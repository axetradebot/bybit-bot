"""
RSI Divergence strategy — enters on regular/hidden RSI divergence
confirmed by a supertrend direction flip and positive EMA slope.

Relaxed slightly to generate more signals (only had 23 trades before)
while maintaining quality:
- Expanded supertrend flip window to 3 bars
- Relaxed StochRSI bands (90/10 instead of 80/20)
- Better R:R ratio (1:3)
"""

from __future__ import annotations

import json
import math
from collections import deque

import pandas as pd

from src.backtest.simulator import build_indicator_snapshot
from src.strategies.base import BaseStrategy, SignalEvent, _sf, _valid


class RSIDivergenceStrategy(BaseStrategy):
    name = "rsi_divergence"

    def __init__(self):
        self._history: deque[dict] = deque(maxlen=12)

    def _snap(self, row: pd.Series) -> dict:
        return {
            "supertrend_dir": _sf(row.get("supertrend_dir")),
            "ema_50": _sf(row.get("ema_50")),
            "low": _sf(row.get("low")),
            "high": _sf(row.get("high")),
        }

    def generate_signal(
        self,
        symbol: str,
        indicators_5m: pd.Series,
        indicators_15m: pd.Series,
        funding_rate: float,
        liq_volume_1h: float,
    ) -> SignalEvent | None:
        curr = indicators_5m
        snap = self._snap(curr)
        self._history.append(snap)

        if len(self._history) < 4:
            return None

        extras = curr.get("extras", {})
        if isinstance(extras, str):
            extras = json.loads(extras)
        if not isinstance(extras, dict):
            extras = {}

        div_reg_bull = extras.get("div_regular_bull", False)
        div_hid_bull = extras.get("div_hidden_bull", False)
        div_reg_bear = extras.get("div_regular_bear", False)
        div_hid_bear = extras.get("div_hidden_bear", False)

        has_bull_div = div_reg_bull or div_hid_bull
        has_bear_div = div_reg_bear or div_hid_bear

        if not has_bull_div and not has_bear_div:
            return None

        # Supertrend flip within last 3 bars
        curr_st = snap["supertrend_dir"]
        bull_flip = False
        bear_flip = False
        for lookback in range(1, min(4, len(self._history))):
            prev_st = self._history[-(lookback + 1)]["supertrend_dir"]
            if curr_st > 0 and prev_st < 0:
                bull_flip = True
            if curr_st < 0 and prev_st > 0:
                bear_flip = True

        # EMA_50 slope
        ema50_now = snap["ema_50"]
        ema50_3ago = (self._history[-4]["ema_50"]
                      if len(self._history) >= 4 else ema50_now)
        ema_slope_up = ema50_now > ema50_3ago
        ema_slope_dn = ema50_now < ema50_3ago

        stochrsi_k = _sf(curr.get("stochrsi_k"))
        rsi = _sf(curr.get("rsi_14"))
        close = _sf(curr.get("close"))
        atr = _sf(curr.get("atr_14"))

        if atr <= 0 or not _valid(curr.get("rsi_14")):
            return None

        window = list(self._history)[-10:]
        swing_low = min(h["low"] for h in window)
        swing_high = max(h["high"] for h in window)

        direction = None

        if has_bull_div and bull_flip and ema_slope_up and stochrsi_k < 90 and rsi > 25:
            direction = "long"
            sl = swing_low - 0.3 * atr
            tp = close + (close - sl) * 2.5
        elif has_bear_div and bear_flip and ema_slope_dn and stochrsi_k > 10 and rsi < 75:
            direction = "short"
            sl = swing_high + 0.3 * atr
            tp = close - (sl - close) * 2.5
        else:
            return None

        # Optional: when CoinGlass liq imbalance is in extras, skip extremes
        imb_v = extras.get("cg_liq_imb")
        if imb_v is not None:
            try:
                imb = float(imb_v)
            except (TypeError, ValueError):
                imb = None
            else:
                if math.isfinite(imb):
                    # Positive = more short-side liquidations (recent squeeze fuel)
                    if direction == "long" and imb > 0.45:
                        return None
                    if direction == "short" and imb < -0.45:
                        return None

        if direction == "long":
            has_regular = div_reg_bull
            has_hidden = div_hid_bull
        else:
            has_regular = div_reg_bear
            has_hidden = div_hid_bear

        conf = 0.65 if has_regular else 0.55
        if has_regular and has_hidden:
            conf += 0.15
        conf = min(conf, 1.0)

        ts = pd.Timestamp(curr.get("timestamp"))
        regime = self._compute_regime(
            _sf(curr.get("atr_pct_rank")), funding_rate, ts,
        )

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
            strategy_combo=["rsi_div", "supertrend_flip", "ema_slope"],
            regime=regime,
            timestamp=ts,
            fill_mode="limit",
        )
        return sig if sig.risk_reward() >= 2.0 else None
