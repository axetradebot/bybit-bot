"""
SignalEvent contract and BaseStrategy ABC.

Every strategy returns a SignalEvent (or None).  The risk manager reads
the signal and either blocks or forwards it to the order manager.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Default partial-take-profit ladder (50 / 30 / 20)
# ---------------------------------------------------------------------------

# Each tuple is (R-multiple of risk distance, fraction of position to close).
# Fractions must sum to <= 1.0; whatever is left runs to the strategy's
# absolute take_profit (or trailing stop / timeout).
DEFAULT_TP_LADDER: tuple[tuple[float, float], ...] = (
    (1.0, 0.50),   # TP1 at 1R closes 50% and triggers BE on the rest
    (2.0, 0.30),   # TP2 at 2R closes another 30%
    (3.0, 0.20),   # TP3 at 3R closes the final 20% (or runner trails)
)


# ---------------------------------------------------------------------------
# Helpers shared by all strategies
# ---------------------------------------------------------------------------

def _sf(val) -> float:
    """Safe float: None / NaN -> 0.0."""
    if val is None:
        return 0.0
    try:
        f = float(val)
        return 0.0 if math.isnan(f) else f
    except (TypeError, ValueError):
        return 0.0


def _valid(val) -> bool:
    """True when *val* is a usable number (not None, not NaN)."""
    if val is None:
        return False
    try:
        return not math.isnan(float(val))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# SignalEvent
# ---------------------------------------------------------------------------

class SignalEvent(BaseModel):
    """Universal signal contract.  Never add exchange-specific fields."""

    symbol: str
    direction: Literal["long", "short", "flat"]
    confidence: float = Field(ge=0.0, le=1.0)
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int = Field(ge=1, le=25)
    position_size_usd: float | None = None
    indicators_snapshot: dict
    strategy_combo: list[str]
    regime: dict
    timestamp: datetime
    timeframe: str = "5m"
    # Backtest / execution hint: limit+maker for mean-reversion, market+taker for momentum.
    # New default is "post_only" (maker-only with TTL fallback in execution layer).
    fill_mode: Literal["limit", "post_only", "market"] | None = None
    risk_pct: float | None = None
    # ─── Per-signal overrides honoured by RiskManager / Simulator ───
    # Strategy-level minimum R:R (lets mean-reversion go down to 1.0
    # without lowering the global gate for everyone).
    min_rr: float | None = None
    # Strategy-level cooldown between consecutive trades, in trading-TF
    # bars. None means "use simulator default (COOLDOWN_BARS)".
    cooldown_bars: int | None = None
    # Stop-loss trigger price kind (live execution).  "MarkPrice" is
    # safer on BTC; "LastPrice" is preferred on illiquid alts where
    # mark/index can drift far from last during cascades.
    sl_trigger_kind: Literal["MarkPrice", "LastPrice", "IndexPrice"] | None = None
    # Partial-TP ladder.  Each (r_multiple, fraction) closes that
    # fraction of the position when price reaches entry ± r * risk.
    # When set, the absolute ``take_profit`` field is the runner /
    # final target.  Fractions must sum to <= 1.0.
    tp_ladder: list[tuple[float, float]] | None = None
    # When TP1 fills, move SL to entry (BE).  Defaults to True when a
    # tp_ladder is present.
    move_be_on_tp1: bool = True

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def _check_tp_ladder(self) -> "SignalEvent":
        if self.tp_ladder is None:
            return self
        total = 0.0
        for r_mult, frac in self.tp_ladder:
            if r_mult <= 0:
                raise ValueError("tp_ladder r_multiple must be > 0")
            if not (0.0 < frac <= 1.0):
                raise ValueError("tp_ladder fraction must be in (0, 1]")
            total += frac
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"tp_ladder fractions sum to {total:.4f} > 1.0",
            )
        return self

    def risk_reward(self) -> float:
        """R:R ratio computed against the runner ``take_profit``."""
        if self.direction == "long":
            reward = self.take_profit - self.entry_price
            risk = self.entry_price - self.stop_loss
        else:
            reward = self.entry_price - self.take_profit
            risk = self.stop_loss - self.entry_price
        return reward / risk if risk > 0 else 0.0

    def effective_min_rr(self, default: float = 1.5) -> float:
        return self.min_rr if self.min_rr is not None else default

    def effective_fill_mode(self, default: str = "post_only") -> str:
        return self.fill_mode if self.fill_mode is not None else default


# ---------------------------------------------------------------------------
# BaseStrategy
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    All strategies inherit from this.

    ``generate_signal()`` is called once per closed 5m bar.
    It receives the current indicator row and must return a ``SignalEvent``
    or ``None`` if no signal conditions are met.

    Subclasses can define ``blocked_regimes`` — a list of
    ``(volatility, funding, session)`` tuples where the strategy should
    NOT trade.  Use ``"*"`` as a wildcard for any dimension.

    The class-level ``min_rr``, ``cooldown_bars``, ``default_tp_ladder``,
    ``default_fill_mode``, and ``default_sl_trigger_kind`` are merged
    onto every emitted ``SignalEvent`` (when the strategy itself does
    not override them on a per-signal basis).  This keeps strategy-
    specific behaviour out of the global risk manager.
    """

    name: str = "base"
    blocked_regimes: list[tuple[str, str, str]] = []
    # Strategy-default execution preferences (used by RiskManager and
    # Simulator when the SignalEvent itself does not set them).
    min_rr: float = 1.5
    cooldown_bars: int = 6              # 6 × 5m = 30 min default
    default_tp_ladder: tuple[tuple[float, float], ...] | None = DEFAULT_TP_LADDER
    default_fill_mode: str = "post_only"
    default_sl_trigger_kind: str = "MarkPrice"
    # When True the strategy is included in regime auto-disable
    # bookkeeping; flip to False for meta-strategies (regime_adaptive)
    # or carry strategies whose performance shouldn't auto-shut them.
    track_for_auto_disable: bool = True

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        indicators_5m: pd.Series,
        indicators_15m: pd.Series,
        funding_rate: float,
        liq_volume_1h: float,
    ) -> SignalEvent | None: ...

    def _regime_allowed(self, regime: dict) -> bool:
        vol = regime["volatility"]
        fund = regime["funding"]
        sess = regime["time_of_day"]
        for bv, bf, bs in self.blocked_regimes:
            if ((bv == "*" or bv == vol) and
                (bf == "*" or bf == fund) and
                (bs == "*" or bs == sess)):
                return False
        return True

    def _compute_regime(
        self,
        atr_pct_rank: float,
        funding_rate: float,
        timestamp: datetime,
    ) -> dict:
        if atr_pct_rank < 0.33:
            vol_regime = "low"
        elif atr_pct_rank < 0.66:
            vol_regime = "medium"
        else:
            vol_regime = "high"

        if funding_rate < -0.0001:
            fund_regime = "negative"
        elif funding_rate > 0.0001:
            fund_regime = "positive"
        else:
            fund_regime = "neutral"

        hour = timestamp.hour
        if hour < 8:
            session = "asia"
        elif hour < 16:
            session = "london"
        elif hour < 22:
            session = "new_york"
        else:
            session = "off_hours"

        return {
            "volatility": vol_regime,
            "funding": fund_regime,
            "time_of_day": session,
        }

    # ------------------------------------------------------------------
    # Helper: stamp class-level defaults onto an emitted signal
    # ------------------------------------------------------------------

    def _finalize_signal(self, sig: SignalEvent) -> SignalEvent | None:
        """
        Apply class defaults (cooldown, fill_mode, tp_ladder, sl_trigger,
        min_rr) to a freshly-built ``SignalEvent`` and run the local
        R:R gate.  Returns ``sig`` or ``None``.

        Strategies should construct a SignalEvent and pass it through
        this helper as the very last step:

            sig = SignalEvent(...)
            return self._finalize_signal(sig)
        """
        if sig is None:
            return None
        if sig.cooldown_bars is None:
            sig.cooldown_bars = self.cooldown_bars
        if sig.fill_mode is None:
            sig.fill_mode = self.default_fill_mode
        if sig.sl_trigger_kind is None:
            sig.sl_trigger_kind = self.default_sl_trigger_kind
        if sig.tp_ladder is None and self.default_tp_ladder is not None:
            sig.tp_ladder = list(self.default_tp_ladder)
        if sig.min_rr is None:
            sig.min_rr = self.min_rr

        if sig.risk_reward() < sig.effective_min_rr():
            return None
        return sig
