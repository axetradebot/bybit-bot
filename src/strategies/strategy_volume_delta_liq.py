"""
Volume Delta + Consolidation Breakout — breakout from tight range
with sustained order flow, volume spike, and trend alignment.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from src.backtest.simulator import build_indicator_snapshot
from src.strategies.base import BaseStrategy, SignalEvent, _sf, _valid


class VolumeDeltaLiqStrategy(BaseStrategy):
    name = "volume_delta_liq"
    blocked_regimes = [
        ("low",    "negative", "*"),
        ("low",    "positive", "asia"),
        ("low",    "neutral",  "new_york"),
        ("low",    "neutral",  "london"),
        ("low",    "positive", "london"),
        ("medium", "negative", "*"),
        ("medium", "positive", "new_york"),
        ("medium", "neutral",  "off_hours"),
        ("high",   "neutral",  "new_york"),
    ]
    min_rr = 2.0
    cooldown_bars = 6
    default_fill_mode = "market"   # breakouts need a taker fill

    # Confidence boost when 1h liquidations exceed the rolling p95.
    # We approximate p95 from a per-symbol rolling window inline since the
    # full DB-backed percentile would mean a query per signal evaluation.
    LIQ_HISTORY_WINDOW = 288       # 24 h of 5-min bars
    LIQ_BOOST_QUANTILE = 0.95
    LIQ_BOOST_AMOUNT = 0.10

    def __init__(self):
        self._history: dict[str, deque[dict]] = {}
        self._vol_history: dict[str, deque[float]] = {}
        self._liq_history: dict[str, deque[float]] = {}

    def _snap(self, row: pd.Series) -> dict:
        return {
            "high": _sf(row.get("high")),
            "low": _sf(row.get("low")),
            "close": _sf(row.get("close")),
            "volume": _sf(row.get("volume")),
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
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=12)
            self._vol_history[symbol] = deque(maxlen=288)
            self._liq_history[symbol] = deque(maxlen=self.LIQ_HISTORY_WINDOW)
        self._history[symbol].append(snap)
        self._vol_history[symbol].append(snap["volume"])
        self._liq_history[symbol].append(max(0.0, _sf(liq_volume_1h)))

        if len(self._history[symbol]) < 6:
            return None

        atr = _sf(c.get("atr_14"))
        ema9 = _sf(c.get("ema_9"))
        ema21 = _sf(c.get("ema_21"))
        ema50 = _sf(c.get("ema_50"))

        if atr <= 0 or not all(_valid(c.get(k)) for k in (
                "atr_14", "ema_9", "ema_21", "ema_50")):
            return None

        atr_rank = _sf(c.get("atr_pct_rank"))
        if atr_rank < 0.18:
            return None

        h = list(self._history[symbol])
        curr = h[-1]
        prev = h[-2]

        # Aggressive flow
        long_flow = curr["order_flow_imb"] > 0.63
        short_flow = curr["order_flow_imb"] < 0.37

        # Volume delta: 2+ of last 3 bars in same direction, current accelerating
        vd = [bar["volume_delta"] for bar in h[-3:]]
        long_delta = sum(1 for v in vd if v > 0) >= 2 and vd[-1] > 0
        short_delta = sum(1 for v in vd if v < 0) >= 2 and vd[-1] < 0

        # Volume spike: 1.8x rolling mean
        vol_spike = False
        if len(self._vol_history[symbol]) >= 30:
            arr = np.array(self._vol_history[symbol])
            avg_vol = float(arr.mean())
            vol_spike = curr["volume"] > 1.8 * avg_vol if avg_vol > 0 else False

        # Tight consolidation: previous 5 bars range < 1.8 ATR
        prev5 = h[-6:-1] if len(h) >= 6 else h[:-1]
        if not prev5:
            return None
        range_high = max(b["high"] for b in prev5)
        range_low = min(b["low"] for b in prev5)
        tight = (range_high - range_low) < 1.6 * atr

        close = curr["close"]
        long_break = close > range_high
        short_break = close < range_low

        # Trend alignment
        long_trend = ema9 > ema21 and ema9 > ema50
        short_trend = ema9 < ema21 and ema9 < ema50

        direction = None

        if long_flow and long_delta and vol_spike and tight and long_break and long_trend:
            direction = "long"
            sl = range_low - 0.3 * atr
            tp = close + (close - sl) * 2.5
        elif short_flow and short_delta and vol_spike and tight and short_break and short_trend:
            direction = "short"
            sl = range_high + 0.3 * atr
            tp = close - (sl - close) * 2.5

        if direction is None:
            return None

        conf = 0.60
        if vol_spike and abs(curr["volume_delta"]) > abs(prev["volume_delta"]):
            conf += 0.15

        # Liquidation cluster boost — when the trailing 24h shows a
        # liquidation flush AND the current bar is *still* elevated,
        # the breakout typically has follow-through fuel.
        liq_now = _sf(liq_volume_1h)
        liq_hist = self._liq_history[symbol]
        if liq_now > 0 and len(liq_hist) >= 30:
            arr = np.array(liq_hist, dtype=float)
            arr = arr[arr > 0]
            if arr.size >= 10:
                p95 = float(np.quantile(arr, self.LIQ_BOOST_QUANTILE))
                if liq_now >= p95:
                    conf += self.LIQ_BOOST_AMOUNT

        conf = min(conf, 1.0)

        ts = pd.Timestamp(c.get("timestamp"))
        regime = self._compute_regime(atr_rank, funding_rate, ts)

        if not self._regime_allowed(regime):
            return None

        sig = SignalEvent(
            symbol=symbol,
            direction=direction,
            confidence=conf,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            leverage=20,
            indicators_snapshot=build_indicator_snapshot(c),
            strategy_combo=["volume_delta", "liq_cluster",
                            "consolidation_break"],
            regime=regime,
            timestamp=ts,
        )
        return self._finalize_signal(sig)
