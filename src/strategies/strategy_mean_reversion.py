"""
Mean-Reversion strategy for choppy/range-bound markets.

Activates when ADX < 20 (opposite of the trend filter) and fades
Bollinger Band extremes with RSI confirmation. Exempt from the chop
filter since it deliberately trades range-bound conditions.

Entry:
  - Long: close below BB lower band, RSI < 30 (oversold), ADX < 20
  - Short: close above BB upper band, RSI > 70 (overbought), ADX < 20
  - BB width must be >= 1.2x ATR (ensures meaningful range)

SL:  Beyond the BB band + 0.3 ATR buffer
TP:  BB mid (VWAP used as secondary target if available)

Position sizing uses 1% risk (smaller than Sniper's 2.5%) since
mean-reversion has lower R:R by nature.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.simulator import build_indicator_snapshot
from src.strategies.base import BaseStrategy, SignalEvent, _sf, _valid


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"
    blocked_regimes: list[tuple[str, str, str]] = [
        ("high", "*", "*"),
    ]

    ADX_CEILING = 20
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    MIN_BB_WIDTH_ATR = 1.2
    SL_BUFFER_ATR = 0.3
    MIN_RR = 1.0

    def generate_signal(
        self,
        symbol: str,
        indicators_5m: pd.Series,
        indicators_15m: pd.Series,
        funding_rate: float,
        liq_volume_1h: float,
    ) -> SignalEvent | None:
        c = indicators_5m

        close = _sf(c.get("close"))
        bb_upper = _sf(c.get("bb_upper"))
        bb_lower = _sf(c.get("bb_lower"))
        bb_mid = _sf(c.get("bb_mid"))
        rsi = _sf(c.get("rsi_14"))
        adx = _sf(c.get("adx_14"))
        atr = _sf(c.get("atr_14"))
        atr_rank = _sf(c.get("atr_pct_rank"))

        needed = ("bb_upper", "bb_lower", "bb_mid", "rsi_14",
                  "adx_14", "atr_14")
        if not all(_valid(c.get(k)) for k in needed):
            return None
        if atr <= 0 or close <= 0:
            return None

        if adx >= self.ADX_CEILING:
            return None

        bb_width = bb_upper - bb_lower
        if bb_width < self.MIN_BB_WIDTH_ATR * atr:
            return None

        direction = None
        sl = tp = 0.0

        if close < bb_lower and rsi < self.RSI_OVERSOLD:
            direction = "long"
            sl = bb_lower - self.SL_BUFFER_ATR * atr
            tp = bb_mid
        elif close > bb_upper and rsi > self.RSI_OVERBOUGHT:
            direction = "short"
            sl = bb_upper + self.SL_BUFFER_ATR * atr
            tp = bb_mid

        if direction is None:
            return None

        vwap = _sf(c.get("vwap"))
        if _valid(c.get("vwap")) and vwap > 0:
            if direction == "long" and vwap < tp:
                tp = vwap
            elif direction == "short" and vwap > tp:
                tp = vwap

        if direction == "long":
            reward = tp - close
            risk = close - sl
        else:
            reward = close - tp
            risk = sl - close

        if risk <= 0:
            return None
        rr = reward / risk
        if rr < self.MIN_RR:
            return None

        conf = 0.50
        if direction == "long" and rsi < 25:
            conf += 0.10
        elif direction == "short" and rsi > 75:
            conf += 0.10

        plus_di = _sf(c.get("plus_di"))
        minus_di = _sf(c.get("minus_di"))
        if plus_di > 0 and minus_di > 0:
            if direction == "long" and plus_di > minus_di:
                conf += 0.10
            elif direction == "short" and minus_di > plus_di:
                conf += 0.10

        mfi = _sf(c.get("mfi"))
        if _valid(c.get("mfi")):
            if direction == "long" and mfi < 25:
                conf += 0.10
            elif direction == "short" and mfi > 75:
                conf += 0.10

        conf = min(conf, 1.0)

        ts = pd.Timestamp(c.get("timestamp"))
        regime = self._compute_regime(atr_rank, funding_rate, ts)

        if not self._regime_allowed(regime):
            return None

        return SignalEvent(
            symbol=symbol,
            direction=direction,
            confidence=conf,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            leverage=20,
            indicators_snapshot=build_indicator_snapshot(c),
            strategy_combo=["mean_reversion", "bb_fade", "rsi_extreme"],
            regime=regime,
            timestamp=ts,
            fill_mode="limit",
        )
