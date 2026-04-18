"""
High Win-Rate v3 — trend-following with proven filters.

Trigger: EMA21 pullback in a trending market.
Edge: 4:1 R:R so each win far exceeds each loss.
       Accept ~23% WR; profit comes from asymmetric payoff.

Backtest results (Apr 2025 - Mar 2026, 5% fixed risk):
  ETH: +$10,799 (+108% return), Sharpe 0.74, 778 trades
       WR 23.3%, payoff 3.41:1, PF 1.03
  SOL: marginal at these params.
  BTC: not profitable with this trigger.

Key settings:
  - risk_pct = 0.05 with --fixed-risk (constant dollar risk)
  - SL = 1.5x ATR, TP = 6x ATR (4:1 R:R)
  - atr_rank >= 0.30 (only trade in volatile conditions)
  - 75%+ of soft filters must confirm

Ablation-proven filters:
  MACD histogram, Supertrend (hard gates).
  VWAP, Heikin Ashi, 15m HTF EMA, MACD accel (soft scoring).
  RSI zone, Near EMA, MFI, OFI removed (hurt WR).
  Trailing stops rejected (exit winners too early).

Fill mode: limit (maker fees = 0.02% vs taker 0.055%).
"""

from __future__ import annotations

import pandas as pd

from src.backtest.simulator import build_indicator_snapshot
from src.strategies.base import BaseStrategy, SignalEvent, _sf, _valid


class HighWinRateStrategy(BaseStrategy):
    name = "high_winrate"
    blocked_regimes: list[tuple[str, str, str]] = []
    # 4 R:R is the strategy's signature — the global 1.5 floor is
    # already cleared trivially.  Override min_rr=4.0 so the rr_gate
    # blocks any degraded setup before it reaches the order manager.
    min_rr = 4.0
    cooldown_bars = 6
    default_fill_mode = "post_only"
    # Asymmetric ladder matched to the 4:1 payoff: protect the win
    # quickly without capping the runner.
    default_tp_ladder = (
        (1.5, 0.30),
        (3.0, 0.30),
        (5.0, 0.20),
    )

    def __init__(self):
        self._prev_close: dict[str, float] = {}
        self._prev_ema21: dict[str, float] = {}
        self._prev_macd: dict[str, float] = {}

    def generate_signal(
        self,
        symbol: str,
        indicators_5m: pd.Series,
        indicators_15m: pd.Series,
        funding_rate: float,
        liq_volume_1h: float,
    ) -> SignalEvent | None:
        c = indicators_5m
        c15 = indicators_15m

        close = _sf(c.get("close"))
        ema21 = _sf(c.get("ema_21"))
        macd_h = _sf(c.get("macd_hist"))

        prev_close = self._prev_close.get(symbol)
        prev_ema21 = self._prev_ema21.get(symbol)
        prev_macd = self._prev_macd.get(symbol)
        self._prev_close[symbol] = close
        self._prev_ema21[symbol] = ema21
        self._prev_macd[symbol] = macd_h

        if prev_close is None or prev_ema21 is None or prev_macd is None:
            return None

        atr = _sf(c.get("atr_14"))
        if atr <= 0 or close <= 0:
            return None

        required = ("ema_9", "ema_21", "ema_50",
                    "macd_hist", "supertrend_dir", "vwap",
                    "ha_close", "ha_open")
        if not all(_valid(c.get(k)) for k in required):
            return None

        ema9 = _sf(c.get("ema_9"))
        ema50 = _sf(c.get("ema_50"))

        # ── Trigger: EMA21 pullback in a trending market ─────────────
        long_trend = ema9 > ema21 > ema50
        short_trend = ema9 < ema21 < ema50

        long_pullback = (prev_close > prev_ema21
                         and close <= ema21 * 1.003)
        short_pullback = (prev_close < prev_ema21
                          and close >= ema21 * 0.997)

        if long_trend and long_pullback:
            direction = "long"
        elif short_trend and short_pullback:
            direction = "short"
        else:
            return None

        # Minimum volatility — only trade in the top 70% of volatility
        atr_rank = _sf(c.get("atr_pct_rank"))
        if atr_rank < 0.30:
            return None

        # ── Ablation-proven filters (hard requirements) ──────────────
        # 1. MACD histogram confirms direction (+1.2% WR avg)
        if direction == "long" and macd_h <= 0:
            return None
        if direction == "short" and macd_h >= 0:
            return None

        # 2. Supertrend direction (+0.3% WR avg)
        st_dir = _sf(c.get("supertrend_dir"))
        if direction == "long" and st_dir <= 0:
            return None
        if direction == "short" and st_dir >= 0:
            return None

        # ── Scoring filters (soft — count confirms) ──────────────────
        score = 0
        n_filters = 0

        # 3. VWAP position (+0.4% WR avg)
        vwap = _sf(c.get("vwap"))
        n_filters += 1
        if (direction == "long" and close > vwap) or \
           (direction == "short" and close < vwap):
            score += 1

        # 4. Heikin Ashi colour (+0.5% WR avg)
        ha_close = _sf(c.get("ha_close"))
        ha_open = _sf(c.get("ha_open"))
        n_filters += 1
        if (direction == "long" and ha_close > ha_open) or \
           (direction == "short" and ha_close < ha_open):
            score += 1

        # 5. 15m HTF EMA alignment (+0.8% WR avg)
        if all(_valid(c15.get(k)) for k in ("ema_9", "ema_21")):
            n_filters += 1
            e9_15 = _sf(c15.get("ema_9"))
            e21_15 = _sf(c15.get("ema_21"))
            if (direction == "long" and e9_15 > e21_15) or \
               (direction == "short" and e9_15 < e21_15):
                score += 1

        # 6. MACD accelerating (momentum building, not waning)
        n_filters += 1
        if (direction == "long" and macd_h > prev_macd) or \
           (direction == "short" and macd_h < prev_macd):
            score += 1

        # 7. No counter-divergence (veto)
        extras = c.get("extras") or {}
        if isinstance(extras, dict):
            if direction == "long" and (extras.get("div_regular_bear") or
                                         extras.get("mom_wave_regular_bear")):
                return None
            if direction == "short" and (extras.get("div_regular_bull") or
                                          extras.get("mom_wave_regular_bull")):
                return None

        # Require >= 75% of soft filters for high conviction
        if n_filters >= 2 and (score / n_filters) < 0.75:
            return None

        # ── Risk: SL = 1.5× ATR, TP = 4× SL = 6× ATR ───────────────
        sl_dist = 1.5 * atr
        tp_dist = 6.0 * atr  # 4:1 R:R (optimizer-proven sweet spot)

        if direction == "long":
            sl = close - sl_dist
            tp = close + tp_dist
        else:
            sl = close + sl_dist
            tp = close - tp_dist

        ts = pd.Timestamp(c.get("timestamp"))
        regime = self._compute_regime(atr_rank, funding_rate, ts)

        conf = (score / n_filters) if n_filters > 0 else 0.5

        sig = SignalEvent(
            symbol=symbol,
            direction=direction,
            confidence=conf,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            leverage=20,
            indicators_snapshot=build_indicator_snapshot(c),
            strategy_combo=["high_winrate", "trend_follow",
                            "ema_pullback"],
            regime=regime,
            timestamp=ts,
        )
        return self._finalize_signal(sig)
