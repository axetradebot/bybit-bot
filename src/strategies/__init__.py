"""
Strategy registry and simulator adapter.

Bridges the ``BaseStrategy.generate_signal()`` interface to the Phase-4
simulator's ``Strategy.on_bar()`` protocol.
"""

from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING

import pandas as pd

from src.backtest.simulator import EntryOrder, build_indicator_snapshot
from src.strategies.strategy_bb_squeeze import BBSqueezeStrategy
from src.strategies.strategy_rsi_divergence import RSIDivergenceStrategy
from src.strategies.strategy_vwap_reversion import VWAPReversionStrategy
from src.strategies.strategy_multitf_scalp import MultiTFScalpStrategy
from src.strategies.strategy_volume_delta_liq import VolumeDeltaLiqStrategy
from src.strategies.strategy_regime_adaptive import RegimeAdaptiveStrategy

if TYPE_CHECKING:
    from src.strategies.base import BaseStrategy

# Maps CLI name -> strategy class
STRATEGY_REGISTRY: dict[str, type] = {
    "bb_squeeze": BBSqueezeStrategy,
    "rsi_divergence": RSIDivergenceStrategy,
    "vwap_reversion": VWAPReversionStrategy,
    "multitf_scalp": MultiTFScalpStrategy,
    "volume_delta_liq": VolumeDeltaLiqStrategy,
    "regime_adaptive": RegimeAdaptiveStrategy,
}

# Mean-reversion: post limit at signal price, maker fees. Momentum: next-bar market, taker fees.
LIMIT_FILL_HEADS: frozenset[str] = frozenset({"vwap_reversion", "rsi_div"})


def _sf(val) -> float:
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


class StrategyAdapter:
    """
    Wraps a ``BaseStrategy`` so it conforms to the Phase-4 simulator's
    ``Strategy`` protocol (``name`` attribute + ``on_bar()`` method).

    Also handles 15m indicator lookup for multi-timeframe strategies.
    """

    def __init__(
        self,
        base: BaseStrategy,
        engine,
        symbol: str,
        from_date: str,
        to_date: str,
    ):
        self.base = base
        self.name = base.name
        self._symbol = symbol

        self._15m_ts: list[pd.Timestamp] = []
        self._15m_rows: list[pd.Series] = []
        self._load_15m(engine, symbol, from_date, to_date)

    def _load_15m(self, engine, symbol: str, from_date: str, to_date: str):
        df = pd.read_sql(
            "SELECT * FROM indicators_15m "
            "WHERE symbol = %(symbol)s "
            "  AND timestamp >= %(from_date)s "
            "  AND timestamp < %(to_date)s "
            "ORDER BY timestamp",
            engine,
            params={"symbol": symbol, "from_date": from_date,
                    "to_date": to_date},
            parse_dates=["timestamp"],
        )
        if df.empty:
            return
        for _, row in df.iterrows():
            self._15m_ts.append(pd.Timestamp(row["timestamp"]))
            self._15m_rows.append(row)

    def _get_15m_row(self, ts: pd.Timestamp) -> pd.Series:
        if not self._15m_ts:
            return pd.Series(dtype=object)
        idx = bisect.bisect_right(self._15m_ts, ts) - 1
        if idx >= 0:
            return self._15m_rows[idx]
        return pd.Series(dtype=object)

    def on_bar(
        self, bar: pd.Series, prev_bar: pd.Series | None,
    ) -> EntryOrder | None:
        ts = pd.Timestamp(bar["timestamp"])
        ind_15m = self._get_15m_row(ts)

        funding = _sf(bar.get("funding_8h"))
        liq = _sf(bar.get("liq_volume_1h"))

        signal = self.base.generate_signal(
            symbol=str(bar.get("symbol", self._symbol)),
            indicators_5m=bar,
            indicators_15m=ind_15m,
            funding_rate=funding,
            liq_volume_1h=liq,
        )

        if signal is None or signal.direction == "flat":
            return None

        fill_mode = signal.fill_mode
        if fill_mode is None:
            head = signal.strategy_combo[0] if signal.strategy_combo else ""
            fill_mode = (
                "limit" if head in LIMIT_FILL_HEADS else "market"
            )

        return EntryOrder(
            direction=signal.direction,
            sl_distance=abs(signal.entry_price - signal.stop_loss),
            tp_distance=abs(signal.take_profit - signal.entry_price),
            strategy_combo=signal.strategy_combo,
            indicators_snapshot=signal.indicators_snapshot,
            limit_price=float(signal.entry_price),
            fill_mode=fill_mode,
        )
