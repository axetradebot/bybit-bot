"""
Multi-timeframe scalp strategy — 15m trend alignment with 5m pullback
entry confirmed by order-flow momentum and fast MACD.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.simulator import build_indicator_snapshot
from src.strategies.base import BaseStrategy, SignalEvent, _sf, _valid


class MultiTFScalpStrategy(BaseStrategy):
    name = "multitf_scalp"

    def __init__(self):
        self._prev_ofi: float | None = None

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
        prev_ofi = self._prev_ofi
        self._prev_ofi = curr_ofi

        if prev_ofi is None:
            return None

        # --- 15m trend alignment ---
        ema9_15 = _sf(c15.get("ema_9"))
        ema21_15 = _sf(c15.get("ema_21"))
        ema50_15 = _sf(c15.get("ema_50"))
        st_15 = _sf(c15.get("supertrend_dir"))

        if not all(_valid(c15.get(k)) for k in ("ema_9", "ema_21", "ema_50",
                                                  "supertrend_dir")):
            return None

        long_15m = ema9_15 > ema21_15 > ema50_15 and st_15 > 0
        short_15m = ema9_15 < ema21_15 < ema50_15 and st_15 < 0

        if not long_15m and not short_15m:
            return None

        # --- 5m conditions ---
        close = _sf(c5.get("close"))
        ema21 = _sf(c5.get("ema_21"))
        ema50 = _sf(c5.get("ema_50"))
        atr = _sf(c5.get("atr_14"))
        rsi = _sf(c5.get("rsi_14"))
        macd_fast_h = _sf(c5.get("macd_fast_hist"))

        if atr <= 0 or not all(_valid(c5.get(k)) for k in (
                "ema_21", "ema_50", "atr_14", "rsi_14", "macd_fast_hist")):
            return None

        pullback_dist = abs(close - ema21)

        direction = None

        if long_15m:
            pullback_ok = pullback_dist <= 0.3 * atr and close >= ema21
            flow_flip = curr_ofi > 0.52 and prev_ofi < 0.48
            rsi_ok = 40 <= rsi <= 65
            macd_ok = macd_fast_h > 0

            if pullback_ok and flow_flip and rsi_ok and macd_ok:
                direction = "long"
                sl = ema50 - 0.5 * atr
                tp = close + (close - sl) * 1.8

        if direction is None and short_15m:
            pullback_ok = pullback_dist <= 0.3 * atr and close <= ema21
            flow_flip = curr_ofi < 0.48 and prev_ofi > 0.52
            rsi_ok = 35 <= rsi <= 60
            macd_ok = macd_fast_h < 0

            if pullback_ok and flow_flip and rsi_ok and macd_ok:
                direction = "short"
                sl = ema50 + 0.5 * atr
                tp = close - (sl - close) * 1.8

        if direction is None:
            return None

        ts = pd.Timestamp(c5.get("timestamp"))
        regime = self._compute_regime(
            _sf(c5.get("atr_pct_rank")), funding_rate, ts,
        )

        sig = SignalEvent(
            symbol=symbol,
            direction=direction,
            confidence=0.65,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            leverage=10,
            indicators_snapshot=build_indicator_snapshot(c5),
            strategy_combo=["multitf_trend", "ema_pullback",
                            "flow_confirmation"],
            regime=regime,
            timestamp=ts,
        )
        return sig if sig.risk_reward() >= 1.5 else None
