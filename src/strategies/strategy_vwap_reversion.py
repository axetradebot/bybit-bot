"""
VWAP Reversion — mean-reversion when price deviates beyond VWAP bands
in low-volatility, funding-neutral conditions.
"""

from __future__ import annotations

import pandas as pd

from src.backtest.simulator import build_indicator_snapshot
from src.strategies.base import BaseStrategy, SignalEvent, _sf, _valid


class VWAPReversionStrategy(BaseStrategy):
    name = "vwap_reversion"
    blocked_regimes = [
        ("medium", "negative", "new_york"),
        ("medium", "positive", "new_york"),
        ("medium", "negative", "asia"),
        # Mean reversion bleeds in high vol (worst USD/trade in backtest slice)
        ("high",   "*",        "*"),
    ]

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
        vwap = _sf(c.get("vwap"))
        vwap_lo1 = _sf(c.get("vwap_dev_lower1"))
        vwap_hi1 = _sf(c.get("vwap_dev_upper1"))
        vwap_lo2 = _sf(c.get("vwap_dev_lower2"))
        vwap_hi2 = _sf(c.get("vwap_dev_upper2"))
        rsi = _sf(c.get("rsi_14"))
        atr_rank = _sf(c.get("atr_pct_rank"))
        atr = _sf(c.get("atr_14"))
        ofi = _sf(c.get("order_flow_imb"))
        squeeze = c.get("bb_squeeze")

        needed = ("vwap", "vwap_dev_lower1", "vwap_dev_upper1",
                  "vwap_dev_lower2", "vwap_dev_upper2", "rsi_14",
                  "atr_pct_rank", "atr_14", "order_flow_imb")
        if not all(_valid(c.get(k)) for k in needed):
            return None
        if atr <= 0:
            return None

        funding_neutral = -0.00015 <= funding_rate <= 0.00015
        not_squeeze = squeeze is not True and squeeze != True  # noqa: E712

        direction = None

        if (close < vwap_lo1 and rsi < 32 and atr_rank < 0.40
                and funding_neutral and not_squeeze and ofi > 0.55):
            direction = "long"
            sl = vwap_lo2 - 0.2 * atr
            tp = vwap
        elif (close > vwap_hi1 and rsi > 68 and atr_rank < 0.40
              and funding_neutral and not_squeeze and ofi < 0.45):
            direction = "short"
            sl = vwap_hi2 + 0.2 * atr
            tp = vwap
        else:
            return None

        conf = 0.60
        if direction == "long" and close < vwap_lo2:
            conf += 0.15
        elif direction == "short" and close > vwap_hi2:
            conf += 0.15
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
            leverage=10,
            indicators_snapshot=build_indicator_snapshot(c),
            strategy_combo=["vwap_reversion", "rsi_filter", "atr_regime",
                            "funding_neutral"],
            regime=regime,
            timestamp=ts,
            fill_mode="limit",
        )
        return sig if sig.risk_reward() >= 1.8 else None
