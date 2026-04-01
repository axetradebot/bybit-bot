"""
Multi-timeframe scalp — 15m EMA stack trend with 5m pullback entry,
supertrend confirmation, and accelerating MACD.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.simulator import build_indicator_snapshot
from src.strategies.base import BaseStrategy, SignalEvent, _sf, _valid


class MultiTFScalpStrategy(BaseStrategy):
    name = "multitf_scalp"
    blocked_regimes = [
        ("medium", "neutral",  "off_hours"),
        ("medium", "neutral",  "asia"),
        ("medium", "positive", "new_york"),
        ("medium", "negative", "london"),
        ("high",   "negative", "asia"),
        ("high",   "positive", "off_hours"),
        ("low",    "neutral",  "off_hours"),
        ("low",    "*",        "new_york"),
        ("low",    "neutral",  "asia"),
    ]

    def __init__(self):
        self._prev_ofi: float | None = None
        self._prev_macd: float | None = None

    def generate_signal(
        self,
        symbol: str,
        indicators_5m: pd.Series,
        indicators_15m: pd.Series,
        funding_rate: float,
        liq_volume_1h: float,
    ) -> SignalEvent | None:
        c5 = indicators_5m
        c15 = indicators_15m

        curr_ofi = _sf(c5.get("order_flow_imb"))
        curr_macd = _sf(c5.get("macd_fast_line"))
        prev_ofi = self._prev_ofi
        prev_macd = self._prev_macd
        self._prev_ofi = curr_ofi
        self._prev_macd = curr_macd

        if prev_ofi is None or prev_macd is None:
            return None

        atr_rank = _sf(c5.get("atr_pct_rank"))
        if atr_rank < 0.15:
            return None

        # 15m: full EMA stack
        ema9_15 = _sf(c15.get("ema_9"))
        ema21_15 = _sf(c15.get("ema_21"))
        ema50_15 = _sf(c15.get("ema_50"))

        if not all(_valid(c15.get(k)) for k in ("ema_9", "ema_21", "ema_50")):
            return None

        long_15m = ema9_15 > ema21_15 > ema50_15
        short_15m = ema9_15 < ema21_15 < ema50_15

        if not long_15m and not short_15m:
            return None

        close = _sf(c5.get("close"))
        ema21 = _sf(c5.get("ema_21"))
        ema50 = _sf(c5.get("ema_50"))
        atr = _sf(c5.get("atr_14"))
        rsi = _sf(c5.get("rsi_14"))
        st_dir = _sf(c5.get("supertrend_dir"))

        needed = ("ema_21", "ema_50", "atr_14", "rsi_14",
                  "macd_fast_line", "supertrend_dir")
        if atr <= 0 or not all(_valid(c5.get(k)) for k in needed):
            return None

        direction = None

        if long_15m:
            near_ema = abs(close - ema21) <= 1.2 * atr
            flow_ok = curr_ofi > 0.58
            rsi_ok = 38 <= rsi <= 68
            macd_accel = curr_macd > 0 and curr_macd > prev_macd
            trend_ok = close > ema50
            st_ok = st_dir > 0

            if near_ema and flow_ok and rsi_ok and macd_accel and trend_ok and st_ok:
                direction = "long"
                sl = close - 1.2 * atr
                tp = close + 2.4 * atr

        if direction is None and short_15m:
            near_ema = abs(close - ema21) <= 1.2 * atr
            flow_ok = curr_ofi < 0.42
            rsi_ok = 32 <= rsi <= 62
            macd_accel = curr_macd < 0 and curr_macd < prev_macd
            trend_ok = close < ema50
            st_ok = st_dir < 0

            if near_ema and flow_ok and rsi_ok and macd_accel and trend_ok and st_ok:
                direction = "short"
                sl = close + 1.2 * atr
                tp = close - 2.4 * atr

        if direction is None:
            return None

        ts = pd.Timestamp(c5.get("timestamp"))
        regime = self._compute_regime(atr_rank, funding_rate, ts)

        if not self._regime_allowed(regime):
            return None

        conf = 0.60
        if abs(curr_ofi - 0.50) > 0.15:
            conf += 0.15
        conf = min(conf, 1.0)

        sig = SignalEvent(
            symbol=symbol,
            direction=direction,
            confidence=conf,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            leverage=10,
            indicators_snapshot=build_indicator_snapshot(c5),
            strategy_combo=["multitf_trend", "ema_pullback",
                            "flow_confirmation"],
            regime=regime,
            timestamp=ts,
            fill_mode="market",
        )
        return sig if sig.risk_reward() >= 1.8 else None
