"""
Volume Delta + Liquidation Cluster strategy — breakout from tight
consolidation driven by sustained aggressive order flow and liquidation
exhaustion.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from src.backtest.simulator import build_indicator_snapshot
from src.strategies.base import BaseStrategy, SignalEvent, _sf, _valid


class VolumeDeltaLiqStrategy(BaseStrategy):
    name = "volume_delta_liq"

    def __init__(self):
        self._history: deque[dict] = deque(maxlen=6)
        self._liq_history: deque[float] = deque(maxlen=8640)

    def _snap(self, row: pd.Series) -> dict:
        return {
            "high": _sf(row.get("high")),
            "low": _sf(row.get("low")),
            "close": _sf(row.get("close")),
            "volume_delta": _sf(row.get("volume_delta")),
            "order_flow_imb": _sf(row.get("order_flow_imb")),
        }

    def generate_signal(
        self,
        symbol: str,
        indicators_5m: pd.Series,
        indicators_15m: pd.Series,
        funding_rate: float,
        liq_volume_1h: float,
    ) -> SignalEvent | None:
        c = indicators_5m
        snap = self._snap(c)
        self._history.append(snap)
        self._liq_history.append(liq_volume_1h)

        if len(self._history) < 6:
            return None

        atr = _sf(c.get("atr_14"))
        ema9 = _sf(c.get("ema_9"))
        ema50 = _sf(c.get("ema_50"))

        if atr <= 0 or not all(_valid(c.get(k)) for k in ("atr_14", "ema_9",
                                                            "ema_50")):
            return None

        h = list(self._history)
        curr = h[-1]
        prev = h[-2]

        # 1. Sustained aggressive flow (current AND previous bar)
        long_flow = curr["order_flow_imb"] > 0.65 and prev["order_flow_imb"] > 0.65
        short_flow = curr["order_flow_imb"] < 0.35 and prev["order_flow_imb"] < 0.35

        # 2. Volume delta increasing for 3 consecutive bars
        vd = [bar["volume_delta"] for bar in h[-3:]]
        long_delta = all(v > 0 for v in vd) and vd[-1] > vd[-2] > vd[-3]
        short_delta = all(v < 0 for v in vd) and vd[-1] < vd[-2] < vd[-3]

        # 3. Liquidation cluster — top 70th percentile of 30-day history
        liq_pct = 0.0
        if len(self._liq_history) >= 20:
            arr = np.array(self._liq_history)
            p70 = float(np.percentile(arr[arr > 0], 70)) if (arr > 0).any() else 0.0
            liq_pct = 1.0 if liq_volume_1h >= p70 and p70 > 0 else 0.0
        liq_ok = liq_pct >= 1.0

        # 4. Tight consolidation: range of last 5 bars < 1.0 ATR
        last5 = h[-5:]
        range_high = max(b["high"] for b in last5)
        range_low = min(b["low"] for b in last5)
        tight = (range_high - range_low) < 1.0 * atr

        # 5. Breakout above/below 5-bar range
        close = curr["close"]
        long_break = close > range_high
        short_break = close < range_low

        # 6. Trend alignment
        long_trend = ema9 > ema50
        short_trend = ema9 < ema50

        direction = None

        if long_flow and long_delta and liq_ok and tight and long_break and long_trend:
            direction = "long"
            sl = range_low - 0.3 * atr
            tp = close + (close - sl) * 2.2
        elif short_flow and short_delta and liq_ok and tight and short_break and short_trend:
            direction = "short"
            sl = range_high + 0.3 * atr
            tp = close - (sl - close) * 2.2

        if direction is None:
            return None

        # Confidence boost for extreme liquidation
        conf = 0.60
        if len(self._liq_history) >= 20:
            arr = np.array(self._liq_history)
            p90 = float(np.percentile(arr[arr > 0], 90)) if (arr > 0).any() else 0.0
            if liq_volume_1h >= p90 > 0:
                conf += 0.15
        conf = min(conf, 1.0)

        ts = pd.Timestamp(c.get("timestamp"))
        regime = self._compute_regime(
            _sf(c.get("atr_pct_rank")), funding_rate, ts,
        )

        sig = SignalEvent(
            symbol=symbol,
            direction=direction,
            confidence=conf,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            leverage=10,
            indicators_snapshot=build_indicator_snapshot(c),
            strategy_combo=["volume_delta", "liq_cluster",
                            "consolidation_break"],
            regime=regime,
            timestamp=ts,
        )
        return sig if sig.risk_reward() >= 1.5 else None
