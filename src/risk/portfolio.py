"""
Portfolio-level risk primitives used by ``RiskManager``.

Each primitive is a small, side-effect-free class with a clear API
that returns either a *multiplier* on risk (sub-1.0 reduces, 1.0 is
neutral) or a *boolean* go/no-go decision.  This keeps RiskManager.evaluate
declarative: gates compose by chaining multipliers.

All DB lookups are wrapped in try/except and silently fall back to a
neutral default — risk components must never crash live trading.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import text as sa_text

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# 1. Confidence-scaled fractional Kelly sizer
# ---------------------------------------------------------------------------

@dataclass
class KellyStats:
    n: int
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float       # Stored as positive number (magnitude)

    def kelly_fraction(self) -> float:
        """Full-Kelly fraction; caller applies a safety multiplier."""
        if self.avg_loss_pct <= 0 or self.n < 5:
            return 0.0
        b = self.avg_win_pct / self.avg_loss_pct
        if b <= 0:
            return 0.0
        f = (b * self.win_rate - (1 - self.win_rate)) / b
        return max(0.0, min(f, 1.0))


class KellySizer:
    """
    Confidence-scaled fractional Kelly sizing.

    Per (strategy_combo, symbol), pulls win rate + avg-win/avg-loss from
    ``trades_log`` (rolling 90 days, live trades only) and computes a
    quarter-Kelly fraction modulated by the signal's confidence.

    The result is a *recommended* ``risk_pct``.  Callers clip into
    ``[risk_floor, risk_ceiling]`` and may further scale by drawdown,
    vol-target, etc.
    """

    SAFETY_MULT = 0.25         # quarter-Kelly is widely accepted as defensive
    LOOKBACK_DAYS = 90
    MIN_TRADES_FOR_KELLY = 20
    CACHE_TTL_S = 300

    def __init__(self, engine: Any | None = None,
                 default_risk_pct: float = 0.025):
        self._engine = engine
        self._default = default_risk_pct
        self._cache: dict[tuple[tuple[str, ...], str], tuple[float, KellyStats]] = {}

    def _query_stats(
        self, combo: list[str], symbol: str,
    ) -> KellyStats | None:
        if self._engine is None:
            return None
        key = (tuple(combo), symbol)
        cached = self._cache.get(key)
        now = time.time()
        if cached and now - cached[0] < self.CACHE_TTL_S:
            return cached[1]

        try:
            with self._engine.connect() as conn:
                row = conn.execute(
                    sa_text(
                        """
                        SELECT
                          COUNT(*) FILTER (WHERE win_loss IS NOT NULL)        AS n,
                          AVG(CASE WHEN win_loss THEN 1.0 ELSE 0.0 END)       AS wr,
                          AVG(CASE WHEN win_loss IS TRUE  THEN pnl_pct END)   AS avg_w,
                          AVG(CASE WHEN win_loss IS FALSE THEN pnl_pct END)   AS avg_l
                        FROM trades_log
                        WHERE strategy_combo = :combo
                          AND symbol         = :symbol
                          AND is_backtest    = FALSE
                          AND entry_time     > NOW() - (:days * INTERVAL '1 day')
                        """,
                    ),
                    {"combo": combo, "symbol": symbol,
                     "days": self.LOOKBACK_DAYS},
                ).first()
        except Exception as exc:
            log.warning("kelly_query_failed", error=str(exc))
            return None
        if row is None:
            return None
        n = int(row[0] or 0)
        wr = float(row[1] or 0.0)
        avg_w = float(row[2] or 0.0)
        avg_l = float(row[3] or 0.0)
        stats = KellyStats(
            n=n, win_rate=wr,
            avg_win_pct=max(avg_w, 0.0),
            avg_loss_pct=abs(min(avg_l, 0.0)),
        )
        self._cache[key] = (now, stats)
        return stats

    def recommended_risk_pct(
        self,
        combo: list[str],
        symbol: str,
        confidence: float,
        risk_floor: float = 0.005,
        risk_ceiling: float = 0.025,
    ) -> float:
        """Return a confidence-scaled, fractional-Kelly-derived ``risk_pct``."""
        stats = self._query_stats(combo, symbol)
        if stats is None or stats.n < self.MIN_TRADES_FOR_KELLY:
            base = self._default
        else:
            base = stats.kelly_fraction() * self.SAFETY_MULT
            if base <= 0:
                base = risk_floor
        # Confidence is in [0, 1]; map (0.4, 1.0) -> (0.5, 1.0) multiplier
        # so a low-confidence signal still trades but at half size.
        c = max(0.4, min(confidence, 1.0))
        conf_mult = 0.5 + (c - 0.4) * (0.5 / 0.6)
        out = base * conf_mult
        return max(risk_floor, min(out, risk_ceiling))


# ---------------------------------------------------------------------------
# 2. Continuous drawdown throttle
# ---------------------------------------------------------------------------

class DrawdownThrottle:
    """
    Replaces the binary 5%-daily-loss circuit-breaker with a continuous
    risk multiplier driven by the current drawdown vs. trailing-30-day
    high-water mark.

    Mapping (linear interpolation between knots):
        DD =  0.00%  -> 1.00 x
        DD =  5.00%  -> 0.70 x
        DD = 10.00%  -> 0.40 x
        DD = 15.00%  -> 0.25 x
        DD >= 20.00% -> hard halt (returns 0.0 multiplier, halt=True)

    The multiplier is applied to the per-trade ``risk_pct``; the hard
    halt mirrors the old daily-loss behaviour but at a more punishing
    threshold (since the throttle has already cut size).
    """

    KNOTS: tuple[tuple[float, float], ...] = (
        (0.00, 1.00),
        (0.05, 0.70),
        (0.10, 0.40),
        (0.15, 0.25),
    )
    HARD_HALT_DD = 0.20

    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        self._peak_equity: float | None = None
        self._peak_set_at: float = 0.0

    def update_peak(self, equity_now: float) -> None:
        if self._peak_equity is None or equity_now > self._peak_equity:
            self._peak_equity = equity_now
            self._peak_set_at = time.time()
        elif (time.time() - self._peak_set_at
              > self.lookback_days * 86_400):
            # Roll the peak down after lookback expires
            self._peak_equity = equity_now
            self._peak_set_at = time.time()

    def multiplier(self, equity_now: float) -> tuple[float, bool]:
        """Return (risk_multiplier, hard_halt_flag)."""
        self.update_peak(equity_now)
        if self._peak_equity is None or self._peak_equity <= 0:
            return 1.0, False
        dd = max(0.0, 1.0 - equity_now / self._peak_equity)
        if dd >= self.HARD_HALT_DD:
            return 0.0, True
        # piecewise-linear interpolation
        knots = self.KNOTS
        if dd <= knots[0][0]:
            return knots[0][1], False
        if dd >= knots[-1][0]:
            return knots[-1][1], False
        for (x0, y0), (x1, y1) in zip(knots, knots[1:]):
            if x0 <= dd <= x1:
                w = (dd - x0) / (x1 - x0)
                return y0 + w * (y1 - y0), False
        return 1.0, False


# ---------------------------------------------------------------------------
# 3. Portfolio correlation gate (β-BTC notional cap)
# ---------------------------------------------------------------------------

class PortfolioCorrelationGate:
    """
    Caps the *implied* BTC-beta of the open-position book + the
    candidate signal.

    Most major USDT-perps run ~0.85-0.95 correlated to BTC.  A naive
    "max 3 simultaneous positions" gate happily lets you stack 3 long
    alts that are all 90 % long-BTC in disguise.  This gate computes
    rolling 7-day β to BTCUSDT and rejects (or downsizes) signals that
    push the total β-notional past a configurable cap.

    When a candidate would breach, we *first* try to scale it down
    (preserving the trade); only if the trade would shrink below
    ``min_position_pct * equity`` do we reject outright.
    """

    LOOKBACK_DAYS = 7
    REFRESH_S = 3600
    MAX_BETA_NOTIONAL_MULT = 1.0    # |β·notional| <= 1.0 × equity
    MIN_POSITION_PCT = 0.10         # don't bother trading <10% of normal

    def __init__(self, engine: Any | None = None,
                 base_symbol: str = "BTCUSDT"):
        self._engine = engine
        self._base = base_symbol
        self._beta_cache: dict[str, float] = {}
        self._cache_ts: float = 0.0

    def _refresh_betas(self, symbols: list[str]) -> None:
        if self._engine is None:
            return
        if time.time() - self._cache_ts < self.REFRESH_S and self._beta_cache:
            return
        try:
            from sqlalchemy import bindparam

            wanted = sorted(set(symbols + [self._base]))
            with self._engine.connect() as conn:
                cutoff = (datetime.utcnow()
                          - timedelta(days=self.LOOKBACK_DAYS))
                stmt = sa_text(
                    "SELECT symbol, timestamp, close FROM candles_5m "
                    "WHERE symbol IN :syms AND timestamp > :cutoff "
                    "ORDER BY timestamp",
                ).bindparams(bindparam("syms", expanding=True))
                df = pd.read_sql(
                    stmt, conn,
                    params={"syms": wanted, "cutoff": cutoff},
                )
        except Exception as exc:
            log.warning("portfolio_beta_refresh_failed", error=str(exc))
            return
        if df.empty or self._base not in df["symbol"].unique():
            return
        # 1h log-returns
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        wide = (df.pivot(index="timestamp", columns="symbol", values="close")
                  .resample("1H").last().pct_change().dropna(how="all"))
        if self._base not in wide.columns or len(wide) < 24:
            return
        base_ret = wide[self._base]
        var_b = float(base_ret.var())
        if var_b <= 0:
            return
        new_betas: dict[str, float] = {self._base: 1.0}
        for sym in wide.columns:
            if sym == self._base:
                continue
            covariance = float(wide[sym].cov(base_ret))
            new_betas[sym] = covariance / var_b
        self._beta_cache = new_betas
        self._cache_ts = time.time()
        log.info("portfolio_beta_refreshed",
                 n_symbols=len(new_betas),
                 sample_btc=new_betas.get(self._base),
                 sample_eth=new_betas.get("ETHUSDT"))

    def _signed_beta_notional(
        self,
        symbol: str,
        notional_usd: float,
        direction: str,
    ) -> float:
        """β · notional, signed by direction (+long, −short)."""
        beta = self._beta_cache.get(symbol, 0.9)
        sign = 1.0 if direction == "long" else -1.0
        return beta * notional_usd * sign

    def evaluate(
        self,
        candidate_symbol: str,
        candidate_notional: float,
        candidate_direction: str,
        equity: float,
        open_positions: list[dict],
    ) -> tuple[bool, float]:
        """
        Returns (allow, scaled_notional).

        - allow=False means even after downscaling, the trade can't fit.
        - allow=True with scaled_notional <= candidate_notional means
          the trade is approved at the (possibly reduced) size.
        """
        symbols = [candidate_symbol] + [
            p.get("symbol", "") for p in open_positions if p.get("symbol")
        ]
        self._refresh_betas(symbols)

        cap = equity * self.MAX_BETA_NOTIONAL_MULT
        existing_beta = 0.0
        for pos in open_positions:
            sym = pos.get("symbol", "")
            notional = abs(float(
                pos.get("notional") or pos.get("position_size_usd") or 0.0,
            ))
            side = pos.get("side") or pos.get("direction") or "long"
            existing_beta += self._signed_beta_notional(sym, notional, side)

        candidate_beta = self._signed_beta_notional(
            candidate_symbol, candidate_notional, candidate_direction,
        )
        total = existing_beta + candidate_beta

        if abs(total) <= cap:
            return True, candidate_notional

        # Downscale candidate to bring |total| to exactly the cap.
        # (Skip if existing book is already past cap — reject outright.)
        if abs(existing_beta) >= cap:
            return False, 0.0
        room = cap - abs(existing_beta) if abs(existing_beta) < cap else 0.0
        # Same-sign exposure: scale down by ratio.
        if math.copysign(1.0, candidate_beta) == math.copysign(1.0, existing_beta or candidate_beta):
            ratio = room / abs(candidate_beta) if candidate_beta else 1.0
        else:
            ratio = (cap + abs(existing_beta)) / abs(candidate_beta)
        ratio = max(0.0, min(ratio, 1.0))
        new_notional = candidate_notional * ratio
        if new_notional < equity * self.MIN_POSITION_PCT * 0.05:
            return False, 0.0
        return True, new_notional


# ---------------------------------------------------------------------------
# 4. Volatility-targeting equity overlay
# ---------------------------------------------------------------------------

class VolatilityTargetOverlay:
    """
    Maintains rolling realized vol of the live equity curve and returns
    a multiplier that drives vol toward an annualised target.

    multiplier = clip(target_vol / realized_vol, 0.5, 1.5)
    """

    REFRESH_S = 600
    MIN_DAYS = 14

    def __init__(self, engine: Any | None = None,
                 target_ann_vol: float = 0.60):
        self._engine = engine
        self._target = target_ann_vol
        self._mult: float = 1.0
        self._last_refresh: float = 0.0

    def _refresh(self) -> None:
        if self._engine is None:
            return
        if time.time() - self._last_refresh < self.REFRESH_S:
            return
        try:
            with self._engine.connect() as conn:
                df = pd.read_sql(
                    sa_text(
                        "SELECT exit_time, pnl_pct FROM trades_log "
                        "WHERE win_loss IS NOT NULL "
                        "  AND is_backtest = FALSE "
                        "  AND exit_time > NOW() - INTERVAL '90 days' "
                        "ORDER BY exit_time",
                    ),
                    conn,
                )
        except Exception as exc:
            log.warning("vol_target_refresh_failed", error=str(exc))
            self._last_refresh = time.time()
            return
        if df.empty or len(df) < 30:
            return
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
        df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors="coerce")
        daily = (df.set_index("exit_time")["pnl_pct"]
                   .resample("1D").sum().dropna())
        if len(daily) < self.MIN_DAYS:
            return
        ann_vol = float(daily.std() * math.sqrt(365))
        if ann_vol <= 0:
            return
        self._mult = float(np.clip(self._target / ann_vol, 0.5, 1.5))
        self._last_refresh = time.time()
        log.info("vol_target_updated",
                 ann_vol=round(ann_vol, 4),
                 mult=round(self._mult, 3))

    def multiplier(self) -> float:
        self._refresh()
        return self._mult


# ---------------------------------------------------------------------------
# 5. Bayesian regime filter
# ---------------------------------------------------------------------------

@dataclass
class RegimeStats:
    n: int
    win_rate: float
    avg_pnl_pct: float

    def posterior_mean_expectancy(
        self,
        prior_n: float = 30.0,
        prior_pnl_pct: float = 0.0,
    ) -> float:
        """Beta-weighted shrink toward a noise (zero-edge) prior."""
        w = self.n / (self.n + prior_n)
        return w * self.avg_pnl_pct + (1 - w) * prior_pnl_pct


class BayesianRegimeFilter:
    """
    Per (strategy_combo, regime cell), shrunk expectancy estimate.
    Used by RiskManager as a *soft* gate: if posterior mean expectancy
    after fees is < 0, the signal is blocked.

    Replaces the brittle hard-coded ``blocked_regimes`` lists by letting
    the data speak — but only after enough trades have accumulated in a
    cell for the posterior to deviate meaningfully from the prior.
    """

    REFRESH_S = 1800
    LOOKBACK_DAYS = 120
    PRIOR_N = 30.0
    # Costs we subtract from posterior expectancy (1.5 R round-trip taker
    # + slippage + funding).  Strategies with avg_pnl_pct below this floor
    # are blocked.
    COST_FLOOR_PCT = 0.0015

    def __init__(self, engine: Any | None = None):
        self._engine = engine
        # cache: (combo_tuple, cell) -> RegimeStats
        self._cache: dict[tuple[tuple[str, ...], tuple[str, str, str]], RegimeStats] = {}
        self._last_refresh: float = 0.0

    def _refresh(self) -> None:
        if self._engine is None:
            return
        if time.time() - self._last_refresh < self.REFRESH_S and self._cache:
            return
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    sa_text(
                        """
                        SELECT strategy_combo,
                               regime_volatility,
                               regime_funding,
                               regime_time_of_day,
                               COUNT(*)                                AS n,
                               AVG(CASE WHEN win_loss THEN 1.0 ELSE 0.0 END) AS wr,
                               AVG(pnl_pct)                            AS avg_pnl
                        FROM trades_log
                        WHERE win_loss IS NOT NULL
                          AND is_backtest = FALSE
                          AND entry_time > NOW() - (:lkb_days * INTERVAL '1 day')
                          AND regime_volatility   IS NOT NULL
                          AND regime_funding      IS NOT NULL
                          AND regime_time_of_day  IS NOT NULL
                        GROUP BY strategy_combo, regime_volatility,
                                 regime_funding,    regime_time_of_day
                        """,
                    ),
                    {"lkb_days": self.LOOKBACK_DAYS},
                ).fetchall()
        except Exception as exc:
            log.warning("regime_filter_refresh_failed", error=str(exc))
            self._last_refresh = time.time()
            return
        new_cache: dict[tuple[tuple[str, ...], tuple[str, str, str]], RegimeStats] = {}
        for r in rows:
            combo = tuple(str(c) for c in (r[0] or []))
            cell = (str(r[1]), str(r[2]), str(r[3]))
            new_cache[(combo, cell)] = RegimeStats(
                n=int(r[4] or 0),
                win_rate=float(r[5] or 0.0),
                avg_pnl_pct=float(r[6] or 0.0),
            )
        self._cache = new_cache
        self._last_refresh = time.time()
        log.info("regime_filter_refreshed", cells=len(new_cache))

    def is_blocked(
        self,
        combo: list[str],
        regime: dict,
    ) -> tuple[bool, float]:
        """Returns (blocked, posterior_expectancy_pct)."""
        self._refresh()
        cell = (
            regime.get("volatility", ""),
            regime.get("funding", ""),
            regime.get("time_of_day", ""),
        )
        stats = self._cache.get((tuple(combo), cell))
        if stats is None or stats.n < 10:
            # Not enough data to block — let it through.
            return False, 0.0
        posterior = stats.posterior_mean_expectancy(self.PRIOR_N)
        return posterior < self.COST_FLOOR_PCT, posterior


# ---------------------------------------------------------------------------
# 6. Strategy auto-disable on rolling negative expectancy
# ---------------------------------------------------------------------------

class StrategyAutoDisable:
    """
    Maintains an in-memory blocklist of strategy names that have shown
    persistently negative expectancy in the last 30 days.

    A strategy is *disabled* when:
        n_30d >= 30  AND  shrunk_expectancy_pct <= -0.001
    A disabled strategy is *re-enabled* when:
        n_7d  >= 10  AND  expectancy_pct_7d >= 0.0005

    Refreshed hourly.  Entirely independent of the per-signal gate, so
    a disabled strategy's signals never even reach the risk manager.
    """

    REFRESH_S = 3600
    DISABLE_N_FLOOR = 30
    DISABLE_EXP_THRESHOLD = -0.001
    REENABLE_N_FLOOR = 10
    REENABLE_EXP_THRESHOLD = 0.0005

    def __init__(self, engine: Any | None = None):
        self._engine = engine
        self._disabled: set[str] = set()
        self._last_refresh: float = 0.0

    def _refresh(self) -> None:
        if self._engine is None:
            return
        if time.time() - self._last_refresh < self.REFRESH_S:
            return
        try:
            with self._engine.connect() as conn:
                rows30 = conn.execute(
                    sa_text(
                        """
                        SELECT strategy_combo[1] AS strat,
                               COUNT(*)          AS n,
                               AVG(pnl_pct)      AS exp_pct
                        FROM trades_log
                        WHERE win_loss IS NOT NULL
                          AND is_backtest = FALSE
                          AND entry_time > NOW() - INTERVAL '30 days'
                        GROUP BY strategy_combo[1]
                        """,
                    ),
                ).fetchall()
                rows7 = conn.execute(
                    sa_text(
                        """
                        SELECT strategy_combo[1] AS strat,
                               COUNT(*)          AS n,
                               AVG(pnl_pct)      AS exp_pct
                        FROM trades_log
                        WHERE win_loss IS NOT NULL
                          AND is_backtest = FALSE
                          AND entry_time > NOW() - INTERVAL '7 days'
                        GROUP BY strategy_combo[1]
                        """,
                    ),
                ).fetchall()
        except Exception as exc:
            log.warning("auto_disable_refresh_failed", error=str(exc))
            self._last_refresh = time.time()
            return

        new_disabled: set[str] = set(self._disabled)
        recent_by_strat = {
            str(r[0]): (int(r[1] or 0), float(r[2] or 0.0)) for r in rows7
        }
        for r in rows30:
            strat = str(r[0])
            n, exp = int(r[1] or 0), float(r[2] or 0.0)
            # Shrink expectancy toward zero with prior n=20
            shrunk = exp * (n / (n + 20.0))
            if (strat not in new_disabled
                    and n >= self.DISABLE_N_FLOOR
                    and shrunk <= self.DISABLE_EXP_THRESHOLD):
                new_disabled.add(strat)
                log.warning("strategy_auto_disabled",
                            strategy=strat, n=n,
                            shrunk_expectancy_pct=round(shrunk, 5))
        # Re-enable check
        for strat in list(new_disabled):
            n7, exp7 = recent_by_strat.get(strat, (0, 0.0))
            if (n7 >= self.REENABLE_N_FLOOR
                    and exp7 >= self.REENABLE_EXP_THRESHOLD):
                new_disabled.discard(strat)
                log.info("strategy_auto_reenabled",
                         strategy=strat, n_7d=n7,
                         exp_pct_7d=round(exp7, 5))

        self._disabled = new_disabled
        self._last_refresh = time.time()

    def is_disabled(self, strategy_name: str) -> bool:
        self._refresh()
        return strategy_name in self._disabled

    @property
    def disabled(self) -> set[str]:
        self._refresh()
        return set(self._disabled)


# ---------------------------------------------------------------------------
# 7. Funding-aware pre-trade expectancy filter
# ---------------------------------------------------------------------------

class FundingExpectancyFilter:
    """
    Estimates the expected funding cost over the strategy's typical hold
    horizon and rejects trades where funding eats > FUNDING_EDGE_FRAC of
    the strategy's typical edge.

    Crucially, this also kills "I'd long an alt with 0.10 %/8h positive
    funding because the chart looks great" trades that have negative net
    expectancy after a few funding windows.
    """

    REFRESH_S = 1800
    LOOKBACK_DAYS = 60
    FUNDING_EDGE_FRAC = 0.5    # block if funding > 50 % of expected edge
    DEFAULT_HOLD_HOURS = 6.0   # used when DB stats are absent

    def __init__(self, engine: Any | None = None):
        self._engine = engine
        self._cache: dict[tuple[str, ...], tuple[float, float]] = {}
        self._last_refresh: float = 0.0

    def _refresh(self) -> None:
        if self._engine is None:
            return
        if time.time() - self._last_refresh < self.REFRESH_S and self._cache:
            return
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(
                    sa_text(
                        """
                        SELECT strategy_combo,
                               AVG(pnl_pct)                                          AS exp_pct,
                               AVG(EXTRACT(EPOCH FROM (exit_time - entry_time))/3600) AS avg_hold_hr
                        FROM trades_log
                        WHERE win_loss IS NOT NULL
                          AND is_backtest = FALSE
                          AND entry_time > NOW() - (:lkb_days * INTERVAL '1 day')
                        GROUP BY strategy_combo
                        """,
                    ),
                    {"lkb_days": self.LOOKBACK_DAYS},
                ).fetchall()
        except Exception as exc:
            log.warning("funding_filter_refresh_failed", error=str(exc))
            self._last_refresh = time.time()
            return
        new_cache: dict[tuple[str, ...], tuple[float, float]] = {}
        for r in rows:
            combo = tuple(str(c) for c in (r[0] or []))
            new_cache[combo] = (
                float(r[1] or 0.0),
                float(r[2] or self.DEFAULT_HOLD_HOURS),
            )
        self._cache = new_cache
        self._last_refresh = time.time()

    def is_blocked(
        self,
        combo: list[str],
        direction: str,
        funding_rate_8h: float,
    ) -> tuple[bool, float]:
        """Returns (blocked, expected_funding_cost_pct)."""
        self._refresh()
        exp_pct, avg_hold = self._cache.get(
            tuple(combo), (0.005, self.DEFAULT_HOLD_HOURS),
        )
        # cost positive => we *pay* funding (long with positive funding,
        # short with negative funding).
        if direction == "long":
            cost = funding_rate_8h * (avg_hold / 8.0)
        else:
            cost = -funding_rate_8h * (avg_hold / 8.0)
        if cost <= 0:
            return False, cost
        if exp_pct <= 0:
            return cost > 0.0005, cost
        return cost > exp_pct * self.FUNDING_EDGE_FRAC, cost


# ---------------------------------------------------------------------------
# Helper: detect proximity to a funding fix (used to close pre-event)
# ---------------------------------------------------------------------------

def minutes_to_next_funding_fix(ts: datetime) -> int:
    """Bybit funding fixes are at 00:00, 08:00, 16:00 UTC."""
    minute = ts.hour * 60 + ts.minute
    next_fix = next(
        (b for b in (0, 480, 960, 1440) if b > minute),
        1440,
    )
    return next_fix - minute
