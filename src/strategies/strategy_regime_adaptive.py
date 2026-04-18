"""
Regime-adaptive meta-strategy — dynamically selects the best strategy
to fire for the **current** ``(volatility, funding, time_of_day)`` cell.

Bug history
-----------
The pre-2026 implementation accepted ``vol_regime`` and ``fund_regime``
arguments but never used them in the SQL query — it always returned
the global top-5 combos by expectancy regardless of regime.  This file
is the corrected version: it queries ``trades_log`` directly with the
regime columns and applies Bayesian shrinkage so cells with very few
trades fall back to a global prior instead of overfitting noise.

Activation
----------
The meta-strategy stays dormant until at least 200 closed (non-blocked,
non-backtest) trades exist in ``trades_log``.  Below that threshold the
posterior estimates are too uncertain to dispatch on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import structlog
from sqlalchemy import Engine, text

from src.strategies.base import BaseStrategy, SignalEvent, _sf

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Tunables — kept module-level so they can be patched by simulations.
# ---------------------------------------------------------------------------

# Minimum total trades across all combos before the meta-strategy fires.
ACTIVATION_TRADE_FLOOR = 200
# Minimum trades inside a (combo, regime cell) for raw stats to dominate
# the prior.  Below this we shrink heavily toward the combo's global mean.
SHRINKAGE_PRIOR_STRENGTH = 30.0
# Minimum **shrunk** expectancy (in fractional return per trade) for a
# combo to be considered.  Negative expectancy combos are never
# dispatched even if they're the best of a bad bunch.
MIN_SHRUNK_EXPECTANCY = 0.0
# Cache the regime → top-combos mapping for this many bars to avoid
# hammering the DB once per signal evaluation.
CACHE_TTL_BARS = 200
# How often (in evaluations) to refresh the global trade count.
TRADE_COUNT_REFRESH_BARS = 500


@dataclass
class _ComboStats:
    combo: tuple[str, ...]
    n: int
    expectancy_pct: float            # raw mean PnL per trade in this cell
    win_rate: float
    global_expectancy_pct: float     # combo's expectancy across all cells
    global_n: int

    def shrunk_expectancy(
        self, prior_strength: float = SHRINKAGE_PRIOR_STRENGTH,
    ) -> float:
        """James–Stein shrinkage toward the combo's own global mean."""
        if self.n <= 0:
            return self.global_expectancy_pct
        w = self.n / (self.n + prior_strength)
        return w * self.expectancy_pct + (1 - w) * self.global_expectancy_pct


class RegimeAdaptiveStrategy(BaseStrategy):
    name = "regime_adaptive"
    # Meta-strategies should not auto-disable based on their own stats —
    # they're already dispatching the *underlying* strategies.
    track_for_auto_disable = False
    # Use the underlying strategy's ladder when one is dispatched.
    default_tp_ladder = None

    def __init__(self, engine: Engine | None = None):
        self._engine = engine
        self._trade_count: int | None = None
        self._eval_counter: int = 0
        # Cache: (vol, fund, sess) -> (refresh_at_eval, list[_ComboStats])
        self._cache: dict[tuple[str, str, str], tuple[int, list[_ComboStats]]] = {}

    # ------------------------------------------------------------------

    def _get_trade_count(self) -> int:
        if self._engine is None:
            return 0
        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM trades_log "
                        "WHERE win_loss IS NOT NULL "
                        "  AND is_backtest = FALSE",
                    ),
                ).scalar()
                return int(row or 0)
        except Exception as exc:
            log.warning("regime_adaptive_count_failed", error=str(exc))
            return 0

    def _query_combo_stats(
        self, vol: str, fund: str, sess: str,
    ) -> list[_ComboStats]:
        """
        Pull per-combo stats for the given regime cell *and* per-combo
        global stats in a single round-trip.  Returns combos sorted by
        shrunk expectancy descending.
        """
        if self._engine is None:
            return []

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """
                        WITH cell AS (
                          SELECT strategy_combo,
                                 COUNT(*)                         AS n,
                                 AVG(pnl_pct)                     AS exp_cell,
                                 AVG(CASE WHEN win_loss THEN 1.0 ELSE 0.0 END)
                                                                  AS win_rate
                          FROM trades_log
                          WHERE win_loss IS NOT NULL
                            AND is_backtest = FALSE
                            AND regime_volatility   = :vol
                            AND regime_funding      = :fund
                            AND regime_time_of_day  = :sess
                          GROUP BY strategy_combo
                        ),
                        global_ AS (
                          SELECT strategy_combo,
                                 COUNT(*)                         AS gn,
                                 AVG(pnl_pct)                     AS exp_global
                          FROM trades_log
                          WHERE win_loss IS NOT NULL
                            AND is_backtest = FALSE
                          GROUP BY strategy_combo
                        )
                        SELECT cell.strategy_combo,
                               cell.n,
                               cell.exp_cell,
                               cell.win_rate,
                               COALESCE(global_.exp_global, 0)    AS exp_global,
                               COALESCE(global_.gn, 0)            AS gn
                        FROM cell
                        LEFT JOIN global_ USING (strategy_combo)
                        ORDER BY cell.n DESC
                        """,
                    ),
                    {"vol": vol, "fund": fund, "sess": sess},
                ).fetchall()
        except Exception as exc:
            log.warning("regime_adaptive_query_failed",
                        vol=vol, fund=fund, sess=sess, error=str(exc))
            return []

        out: list[_ComboStats] = []
        for r in rows:
            combo_raw = r[0] or []
            combo = tuple(str(c) for c in combo_raw)
            if not combo:
                continue
            try:
                stats = _ComboStats(
                    combo=combo,
                    n=int(r[1] or 0),
                    expectancy_pct=float(r[2] or 0.0),
                    win_rate=float(r[3] or 0.0),
                    global_expectancy_pct=float(r[4] or 0.0),
                    global_n=int(r[5] or 0),
                )
            except (TypeError, ValueError):
                continue
            out.append(stats)

        out.sort(key=lambda s: s.shrunk_expectancy(), reverse=True)
        return out

    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        indicators_5m: pd.Series,
        indicators_15m: pd.Series,
        funding_rate: float,
        liq_volume_1h: float,
    ) -> SignalEvent | None:
        self._eval_counter += 1

        if (self._trade_count is None
                or self._eval_counter % TRADE_COUNT_REFRESH_BARS == 0):
            self._trade_count = self._get_trade_count()

        if (self._trade_count or 0) < ACTIVATION_TRADE_FLOOR:
            return None

        atr_rank = _sf(indicators_5m.get("atr_pct_rank"))
        ts = pd.Timestamp(indicators_5m.get("timestamp"))
        regime = self._compute_regime(atr_rank, funding_rate, ts)
        cell_key = (
            regime["volatility"], regime["funding"], regime["time_of_day"],
        )

        cached = self._cache.get(cell_key)
        if cached is None or self._eval_counter >= cached[0]:
            stats = self._query_combo_stats(*cell_key)
            self._cache[cell_key] = (
                self._eval_counter + CACHE_TTL_BARS, stats,
            )
        else:
            stats = cached[1]

        if not stats:
            return None

        # Lazy-import to avoid a registry cycle at module load time.
        from src.strategies import STRATEGY_REGISTRY

        best_signal: SignalEvent | None = None
        best_score = -math.inf

        for s in stats:
            shrunk = s.shrunk_expectancy()
            if shrunk <= MIN_SHRUNK_EXPECTANCY:
                continue
            head = s.combo[0]
            cls = STRATEGY_REGISTRY.get(head)
            if cls is None or head == self.name:
                continue
            try:
                strat = cls() if head != "regime_adaptive" else cls(engine=self._engine)
            except Exception:
                continue
            sig = strat.generate_signal(
                symbol, indicators_5m, indicators_15m,
                funding_rate, liq_volume_1h,
            )
            if sig is None or sig.direction == "flat":
                continue

            # Score = shrunk expectancy × confidence × log(1 + n) so
            # we mildly prefer combos with deeper sample support.
            score = shrunk * sig.confidence * math.log1p(s.n)
            if score > best_score:
                best_score = score
                # Tag the signal so trades_log captures the meta layer.
                sig.strategy_combo = list(sig.strategy_combo) + ["regime_adaptive"]
                best_signal = sig

        if best_signal is None:
            return None
        # Run the standard finalize step (ladder/cooldown/min_rr defaults
        # come from the underlying strategy because best_signal is the
        # underlying strategy's emission; we just enforce the gate).
        if best_signal.risk_reward() < best_signal.effective_min_rr():
            return None
        return best_signal
