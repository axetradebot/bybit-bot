"""
BB Squeeze — fires on Bollinger-inside-Keltner squeeze release
with EMA trend, MACD momentum, supertrend, and flow confirmation.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.simulator import build_indicator_snapshot
from src.strategies.base import BaseStrategy, SignalEvent, _sf, _valid


class BBSqueezeStrategy(BaseStrategy):
    name = "bb_squeeze"
    blocked_regimes = [
        ("low",    "positive", "asia"),
        ("medium", "positive", "off_hours"),
        ("medium", "positive", "new_york"),
        ("medium", "negative", "london"),
        ("low",    "neutral",  "off_hours"),
        ("medium", "neutral",  "off_hours"),
        ("high",   "negative", "asia"),
        ("low",    "positive", "off_hours"),
    ]

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

        if not (prev["bb_squeeze"] == True and curr.get("bb_squeeze") == False):  # noqa: E712
            return None

        atr = _sf(curr.get("atr_14"))
        if atr <= 0:
            return None

        atr_rank = _sf(curr.get("atr_pct_rank"))
        if atr_rank < 0.25:
            return None

        # BB expansion > 0.35 ATR (stronger release)
        curr_bb_w = _sf(curr.get("bb_upper")) - _sf(curr.get("bb_lower"))
        prev_bb_w = prev["bb_upper"] - prev["bb_lower"]
        if curr_bb_w - prev_bb_w <= 0.35 * atr:
            return None

        ema9 = _sf(curr.get("ema_9"))
        ema21 = _sf(curr.get("ema_21"))
        ema50 = _sf(curr.get("ema_50"))
        macd_h = _sf(curr.get("macd_hist"))
        ofi = _sf(curr.get("order_flow_imb"))
        close = _sf(curr.get("close"))
        vwap = _sf(curr.get("vwap"))
        st_dir = _sf(curr.get("supertrend_dir"))

        if not all(_valid(curr.get(c)) for c in ("ema_9", "ema_21", "ema_50",
                                                   "macd_hist", "order_flow_imb",
                                                   "vwap", "supertrend_dir")):
            return None

        prev_macd = prev["macd_hist"]

        long_trend = ema9 > ema21 > ema50
        short_trend = ema9 < ema21 < ema50
        long_macd = macd_h > 0 and macd_h > prev_macd
        short_macd = macd_h < 0 and macd_h < prev_macd
        long_flow = ofi > 0.60
        short_flow = ofi < 0.40
        long_vwap = close > vwap
        short_vwap = close < vwap
        long_st = st_dir > 0
        short_st = st_dir < 0

        if long_trend and long_macd and long_flow and long_vwap and long_st:
            direction = "long"
            sl = prev["low"] - 0.5 * atr
            tp = close + (close - sl) * 2.5
        elif short_trend and short_macd and short_flow and short_vwap and short_st:
            direction = "short"
            sl = prev["high"] + 0.5 * atr
            tp = close - (sl - close) * 2.5
        else:
            return None

        conf = 0.55
        ema200 = _sf(curr.get("ema_200"))
        if _valid(curr.get("ema_200")):
            if (direction == "long" and ema50 > ema200) or \
               (direction == "short" and ema50 < ema200):
                conf += 0.15
        rsi = _sf(curr.get("rsi_14"))
        if 40 < rsi < 60:
            conf += 0.10
        conf = min(conf, 1.0)

        ts = pd.Timestamp(curr.get("timestamp"))
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
            leverage=10,
            indicators_snapshot=build_indicator_snapshot(curr),
            strategy_combo=["bb_squeeze", "ema_trend", "macd_momentum", "vwap_filter"],
            regime=regime,
            timestamp=ts,
            fill_mode="market",
        )
        return sig if sig.risk_reward() >= 2.0 else None
