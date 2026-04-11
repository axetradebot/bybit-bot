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
from pydantic import BaseModel, Field


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
    # Backtest / execution hint: limit+maker for mean-reversion, market+taker for momentum
    fill_mode: Literal["limit", "market"] | None = None
    risk_pct: float | None = None

    model_config = {"arbitrary_types_allowed": True}

    def risk_reward(self) -> float:
        """R:R ratio.  Must be >= 1.5 or the risk manager will block."""
        if self.direction == "long":
            reward = self.take_profit - self.entry_price
            risk = self.entry_price - self.stop_loss
        else:
            reward = self.entry_price - self.take_profit
            risk = self.stop_loss - self.entry_price
        return reward / risk if risk > 0 else 0.0


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
    """

    name: str = "base"
    blocked_regimes: list[tuple[str, str, str]] = []

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
