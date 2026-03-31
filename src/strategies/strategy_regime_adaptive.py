"""
Regime-adaptive meta-strategy — dynamically selects the highest-scoring
strategy based on recent strategy_performance for the current market regime.

Only activates when trades_log has >= 500 rows.
"""

from __future__ import annotations

import pandas as pd
import structlog
from sqlalchemy import Engine, text

from src.backtest.simulator import build_indicator_snapshot
from src.strategies.base import BaseStrategy, SignalEvent, _sf, _valid

log = structlog.get_logger()


class RegimeAdaptiveStrategy(BaseStrategy):
    name = "regime_adaptive"

    def __init__(self, engine: Engine | None = None):
        self._engine = engine
        self._trade_count: int | None = None
        self._checked_at: int = 0

    def _get_trade_count(self) -> int:
        if self._engine is None:
            return 0
        with self._engine.connect() as conn:
            row = conn.execute(
                text("SELECT COUNT(*) FROM trades_log")
            ).scalar()
            return int(row or 0)

    def _query_top_combos(self, vol_regime: str, fund_regime: str):
        if self._engine is None:
            return []
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT strategy_combo, win_rate, sharpe
                    FROM strategy_performance
                    WHERE total_trades >= 200
                      AND win_rate >= 0.58
                      AND regime_funding = :fund
                      AND last_updated >= NOW() - INTERVAL '24 hours'
                    ORDER BY win_rate * sharpe DESC
                    LIMIT 3
                """),
                {"fund": fund_regime},
            ).fetchall()
        return rows

    def generate_signal(
        self,
        symbol: str,
        indicators_5m: pd.Series,
        indicators_15m: pd.Series,
        funding_rate: float,
        liq_volume_1h: float,
    ) -> SignalEvent | None:
        self._checked_at += 1

        # Refresh trade count every 500 bars to avoid constant DB hits
        if self._trade_count is None or self._checked_at % 500 == 0:
            self._trade_count = self._get_trade_count()

        if self._trade_count < 500:
            return None

        atr_rank = _sf(indicators_5m.get("atr_pct_rank"))
        ts = pd.Timestamp(indicators_5m.get("timestamp"))
        regime = self._compute_regime(atr_rank, funding_rate, ts)

        top = self._query_top_combos(regime["volatility"], regime["funding"])
        if not top:
            return None

        # Dynamically instantiate the top strategy by name
        from src.strategies import STRATEGY_REGISTRY
        best_signal: SignalEvent | None = None
        best_score = -1.0

        for row in top:
            combo = row[0]
            win_rate = float(row[1])
            strat_name = combo[0] if combo else None
            if strat_name is None or strat_name not in STRATEGY_REGISTRY:
                continue

            strat_cls = STRATEGY_REGISTRY[strat_name]
            strat = strat_cls()
            sig = strat.generate_signal(
                symbol, indicators_5m, indicators_15m,
                funding_rate, liq_volume_1h,
            )
            if sig is None or sig.direction == "flat":
                continue

            score = sig.confidence * win_rate
            if score > best_score:
                best_score = score
                best_signal = sig

        if best_signal is None:
            return None

        best_signal.confidence = min(best_score, 1.0)
        return best_signal if best_signal.risk_reward() >= 1.5 else None
