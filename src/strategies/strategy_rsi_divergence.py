"""
RSI Divergence strategy — enters on regular/hidden RSI divergence
confirmed by a supertrend direction flip and positive EMA slope.
"""

from __future__ import annotations

import json
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
            "extras": row.get("extras"),
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

        # --- divergence flags from extras JSONB ---
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

        # --- supertrend flip (current or previous bar) ---
        curr_st = snap["supertrend_dir"]
        prev_st = self._history[-2]["supertrend_dir"]

        bull_flip = (curr_st > 0 and prev_st < 0) or \
                    (len(self._history) >= 3 and
                     self._history[-2]["supertrend_dir"] > 0 and
                     self._history[-3]["supertrend_dir"] < 0)
        bear_flip = (curr_st < 0 and prev_st > 0) or \
                    (len(self._history) >= 3 and
                     self._history[-2]["supertrend_dir"] < 0 and
                     self._history[-3]["supertrend_dir"] > 0)

        # --- EMA_50 slope (compare to 3 bars ago) ---
        ema50_now = snap["ema_50"]
        ema50_3ago = self._history[-4]["ema_50"] if len(self._history) >= 4 else ema50_now
        ema_slope_up = ema50_now > ema50_3ago
        ema_slope_dn = ema50_now < ema50_3ago

        stochrsi_k = _sf(curr.get("stochrsi_k"))
        rsi = _sf(curr.get("rsi_14"))
        close = _sf(curr.get("close"))
        atr = _sf(curr.get("atr_14"))

        if atr <= 0 or not _valid(curr.get("rsi_14")):
            return None

        # --- swing low/high over last 10 bars ---
        window = list(self._history)[-10:]
        swing_low = min(h["low"] for h in window)
        swing_high = max(h["high"] for h in window)

        direction = None

        if has_bull_div and bull_flip and ema_slope_up and stochrsi_k < 80 and rsi > 30:
            direction = "long"
            sl = swing_low - 0.3 * atr
            tp = close + (close - sl) * 2.5
        elif has_bear_div and bear_flip and ema_slope_dn and stochrsi_k > 20 and rsi < 70:
            direction = "short"
            sl = swing_high + 0.3 * atr
            tp = close - (sl - close) * 2.5
        else:
            return None

        # --- confidence ---
        if direction == "long":
            has_regular = div_reg_bull
            has_hidden = div_hid_bull
        else:
            has_regular = div_reg_bear
            has_hidden = div_hid_bear

        conf = 0.6 if has_regular else 0.5
        if has_regular and has_hidden:
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
            strategy_combo=["rsi_div", "supertrend_flip", "ema_slope"],
            regime=regime,
            timestamp=ts,
        )
        return sig if sig.risk_reward() >= 1.5 else None
