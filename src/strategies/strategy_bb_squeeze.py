"""
BB Squeeze strategy — fires on Bollinger-inside-Keltner squeeze release
with EMA trend alignment, MACD momentum, and VWAP/flow confirmation.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.simulator import build_indicator_snapshot
from src.strategies.base import BaseStrategy, SignalEvent, _sf, _valid


class BBSqueezeStrategy(BaseStrategy):
    name = "bb_squeeze"

    def __init__(self):
        self._prev: dict | None = None

    def generate_signal(
        self,
        symbol: str,
        indicators_5m: pd.Series,
        indicators_15m: pd.Series,
        funding_rate: float,
        liq_volume_1h: float,
    ) -> SignalEvent | None:
        curr = indicators_5m
        prev = self._prev
        self._prev = {
            "bb_squeeze": curr.get("bb_squeeze"),
            "bb_upper": _sf(curr.get("bb_upper")),
            "bb_lower": _sf(curr.get("bb_lower")),
            "macd_hist": _sf(curr.get("macd_hist")),
            "low": _sf(curr.get("low")),
            "high": _sf(curr.get("high")),
        }

        if prev is None:
            return None

        # --- core: squeeze release ---
        if not (prev["bb_squeeze"] == True and curr.get("bb_squeeze") == False):  # noqa: E712
            return None

        atr = _sf(curr.get("atr_14"))
        if atr <= 0:
            return None

        # BB width expansion > 0.2 ATR
        curr_bb_w = _sf(curr.get("bb_upper")) - _sf(curr.get("bb_lower"))
        prev_bb_w = prev["bb_upper"] - prev["bb_lower"]
        if curr_bb_w - prev_bb_w <= 0.2 * atr:
            return None

        ema9 = _sf(curr.get("ema_9"))
        ema21 = _sf(curr.get("ema_21"))
        ema50 = _sf(curr.get("ema_50"))
        macd_h = _sf(curr.get("macd_hist"))
        ofi = _sf(curr.get("order_flow_imb"))
        close = _sf(curr.get("close"))
        vwap = _sf(curr.get("vwap"))

        if not all(_valid(curr.get(c)) for c in ("ema_9", "ema_21", "ema_50",
                                                   "macd_hist", "order_flow_imb",
                                                   "vwap")):
            return None

        prev_macd = prev["macd_hist"]

        # --- determine direction ---
        long_trend = ema9 > ema21 > ema50
        short_trend = ema9 < ema21 < ema50
        long_macd = macd_h > 0 and macd_h > prev_macd
        short_macd = macd_h < 0 and macd_h < prev_macd
        long_flow = ofi > 0.55
        short_flow = ofi < 0.45
        long_vwap = close > vwap
        short_vwap = close < vwap

        if long_trend and long_macd and long_flow and long_vwap:
            direction = "long"
            sl = prev["low"] - 0.5 * atr
            tp = close + (close - sl) * 2.0
        elif short_trend and short_macd and short_flow and short_vwap:
            direction = "short"
            sl = prev["high"] + 0.5 * atr
            tp = close - (sl - close) * 2.0
        else:
            return None

        # --- confidence ---
        conf = 0.5
        ema200 = _sf(curr.get("ema_200"))
        if _valid(curr.get("ema_200")):
            if (direction == "long" and ema50 > ema200) or \
               (direction == "short" and ema50 < ema200):
                conf += 0.1
        st_dir = _sf(curr.get("supertrend_dir"))
        if (direction == "long" and st_dir > 0) or \
           (direction == "short" and st_dir < 0):
            conf += 0.1
        rsi = _sf(curr.get("rsi_14"))
        if 40 < rsi < 60:
            conf += 0.1
        conf = min(conf, 1.0)

        ts = pd.Timestamp(curr.get("timestamp"))
        regime = self._compute_regime(
            _sf(curr.get("atr_pct_rank")), funding_rate, ts,
        )

        sig = SignalEvent(
            symbol=symbol,
            direction=direction,
            confidence=conf,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            leverage=10,
            indicators_snapshot=build_indicator_snapshot(curr),
            strategy_combo=["bb_squeeze", "ema_trend", "macd_momentum", "vwap_filter"],
            regime=regime,
            timestamp=ts,
        )
        return sig if sig.risk_reward() >= 1.5 else None
