"""
Sniper Strategy -- ultra-selective multi-timeframe trend continuation.

Philosophy
----------
Only trade when the stars align across BOTH a higher "context" timeframe
and the trading timeframe.  Each (symbol, TF) pair fires rarely, but
spreading across many pairs builds consistent volume.

Trigger: EMA-21 rejection candle
  * Price was above EMA-21 on the previous bar.
  * Current bar dips to (or through) EMA-21 then closes back above it.
  * This confirms EMA-21 acted as support (long) / resistance (short).

Confluence gates (all must pass):
  Context TF  -- EMA-9 > EMA-21 alignment + Supertrend confirms
  Trading TF  -- Full EMA stack (9 > 21 > 50)
              -- EMA spread > 0.3 %  (trend well-established)
              -- EMA-21 rejection candle
              -- RSI 33-65 (long) / 32-68 (short)  (healthy pullback)
              -- Heikin-Ashi reversal candle
              -- Supertrend direction
              -- ATR-rank >= 0.25

Risk: SL = sl_atr_mult * ATR,  TP = tp_atr_mult * ATR  (default 5:1 R:R).
Fill: limit order (maker fees).

All numeric thresholds are configurable via __init__ kwargs so the
optimizer can grid-search over them without touching the code.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.simulator import build_indicator_snapshot
from src.strategies.base import BaseStrategy, SignalEvent, _sf, _valid


class SniperStrategy(BaseStrategy):
    name = "sniper"
    blocked_regimes: list[tuple[str, str, str]] = []

    def __init__(
        self,
        *,
        ema_spread_min: float = 0.003,
        rsi_long_lo: float = 33,
        rsi_long_hi: float = 65,
        rsi_short_lo: float = 32,
        rsi_short_hi: float = 68,
        atr_rank_floor: float = 0.25,
        sl_atr_mult: float = 1.2,
        tp_atr_mult: float = 6.0,
        ema_touch_slack: float = 0.004,
    ):
        self.ema_spread_min = ema_spread_min
        self.rsi_long_lo = rsi_long_lo
        self.rsi_long_hi = rsi_long_hi
        self.rsi_short_lo = rsi_short_lo
        self.rsi_short_hi = rsi_short_hi
        self.atr_rank_floor = atr_rank_floor
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.ema_touch_slack = ema_touch_slack
        self._prev_close: float | None = None
        self._prev_ema21: float | None = None

    def generate_signal(
        self,
        symbol: str,
        indicators_5m: pd.Series,
        indicators_15m: pd.Series,
        funding_rate: float,
        liq_volume_1h: float,
    ) -> SignalEvent | None:
        c = indicators_5m        # trading-TF bar
        ctx = indicators_15m     # context (higher) TF bar

        close = _sf(c.get("close"))
        high = _sf(c.get("high"))
        low = _sf(c.get("low"))

        ema9 = _sf(c.get("ema_9"))
        ema21 = _sf(c.get("ema_21"))
        ema50 = _sf(c.get("ema_50"))
        atr = _sf(c.get("atr_14"))

        prev_close = self._prev_close
        prev_ema21 = self._prev_ema21
        self._prev_close = close
        self._prev_ema21 = ema21

        if prev_close is None or prev_ema21 is None:
            return None
        if atr <= 0 or close <= 0 or ema21 <= 0:
            return None

        required = ("ema_9", "ema_21", "ema_50", "rsi_14",
                     "supertrend_dir", "ha_close", "ha_open")
        if not all(_valid(c.get(k)) for k in required):
            return None

        # ── Context TF: higher-TF trend confirmation ───────
        if not (_valid(ctx.get("ema_9")) and _valid(ctx.get("ema_21"))):
            return None

        ctx_ema9 = _sf(ctx.get("ema_9"))
        ctx_ema21 = _sf(ctx.get("ema_21"))
        ctx_st = _sf(ctx.get("supertrend_dir"))

        ctx_long = ctx_ema9 > ctx_ema21 and ctx_st > 0
        ctx_short = ctx_ema9 < ctx_ema21 and ctx_st < 0
        if not (ctx_long or ctx_short):
            return None

        # ── Trading TF: full EMA stack ─────────────────────
        long_stack = ema9 > ema21 > ema50
        short_stack = ema9 < ema21 < ema50

        if ctx_long and long_stack:
            direction = "long"
        elif ctx_short and short_stack:
            direction = "short"
        else:
            return None

        # ── EMA separation -- trend must be established ────
        ema_spread = abs(ema9 - ema50) / ema50
        if ema_spread < self.ema_spread_min:
            return None

        # ── Trigger: EMA-21 rejection candle ───────────────
        slack_hi = 1.0 + self.ema_touch_slack
        slack_lo = 1.0 - self.ema_touch_slack
        if direction == "long":
            if prev_close <= prev_ema21:
                return None
            if not (low <= ema21 * slack_hi and close > ema21):
                return None
        else:
            if prev_close >= prev_ema21:
                return None
            if not (high >= ema21 * slack_lo and close < ema21):
                return None

        # ── RSI: healthy pullback zone ─────────────────────
        rsi = _sf(c.get("rsi_14"))
        if direction == "long" and not (self.rsi_long_lo <= rsi <= self.rsi_long_hi):
            return None
        if direction == "short" and not (self.rsi_short_lo <= rsi <= self.rsi_short_hi):
            return None

        # ── Heikin-Ashi confirms reversal ──────────────────
        ha_close = _sf(c.get("ha_close"))
        ha_open = _sf(c.get("ha_open"))
        if direction == "long" and ha_close <= ha_open:
            return None
        if direction == "short" and ha_close >= ha_open:
            return None

        # ── Supertrend on trading TF ───────────────────────
        st_dir = _sf(c.get("supertrend_dir"))
        if direction == "long" and st_dir <= 0:
            return None
        if direction == "short" and st_dir >= 0:
            return None

        # ── Minimum volatility ─────────────────────────────
        atr_rank = _sf(c.get("atr_pct_rank"))
        if atr_rank < self.atr_rank_floor:
            return None

        # ── Risk: SL / TP via ATR multiples ────────────────
        sl_dist = self.sl_atr_mult * atr
        tp_dist = self.tp_atr_mult * atr

        if direction == "long":
            sl = close - sl_dist
            tp = close + tp_dist
        else:
            sl = close + sl_dist
            tp = close - tp_dist

        ts = pd.Timestamp(c.get("timestamp"))
        regime = self._compute_regime(atr_rank, funding_rate, ts)

        return SignalEvent(
            symbol=symbol,
            direction=direction,
            confidence=0.8,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            leverage=20,
            indicators_snapshot=build_indicator_snapshot(c),
            strategy_combo=["sniper", "ema_rejection", "multi_tf"],
            regime=regime,
            timestamp=ts,
            fill_mode="limit",
        )
