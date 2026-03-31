"""
Bar-by-bar backtest simulator.

Walks 5m candles sequentially, evaluates a pluggable strategy on each bar,
simulates realistic fills (next-bar open), and tracks funding payments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import create_engine

log = structlog.get_logger()

# All indicators_5m columns that go into the snapshot
INDICATOR_COLS = [
    "ema_9", "ema_21", "ema_50", "ema_200",
    "rsi_14", "stochrsi_k", "stochrsi_d",
    "macd_line", "macd_signal", "macd_hist",
    "macd_fast_line", "macd_fast_signal", "macd_fast_hist",
    "bb_upper", "bb_mid", "bb_lower", "bb_width",
    "kc_upper", "kc_lower", "bb_squeeze",
    "atr_14", "atr_pct_rank",
    "supertrend", "supertrend_dir",
    "vwap", "vwap_dev_upper1", "vwap_dev_lower1",
    "vwap_dev_upper2", "vwap_dev_lower2",
    "obv", "volume_delta", "order_flow_imb",
    "ha_open", "ha_high", "ha_low", "ha_close",
    "funding_8h", "funding_24h_cum", "liq_volume_1h",
    "extras",
]

CANDLE_SNAPSHOT_COLS = [
    "open", "high", "low", "close", "volume",
    "buy_volume", "sell_volume",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EntryOrder:
    direction: str
    sl_distance: float
    tp_distance: float
    strategy_combo: list[str]
    indicators_snapshot: dict


@dataclass
class ClosedTrade:
    symbol: str
    direction: str
    leverage: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    position_size_usd: float
    pnl_pct: float
    pnl_usd: float
    funding_paid_usd: float
    win_loss: bool
    strategy_combo: list[str]
    indicators_snapshot: dict
    regime_volatility: str
    regime_funding: str
    regime_time_of_day: str
    exit_reason: str
    notes: str | None = None


@dataclass
class _OpenPosition:
    direction: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_usd: float
    strategy_combo: list[str]
    indicators_snapshot: dict
    regime_volatility: str
    regime_funding: str
    regime_time_of_day: str
    entry_bar_idx: int
    funding_paid_usd: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_indicator_snapshot(bar: pd.Series) -> dict:
    """Capture every indicator + candle column into a JSON-safe dict."""
    snap: dict = {}
    for col in INDICATOR_COLS + CANDLE_SNAPSHOT_COLS:
        val = bar.get(col)
        if val is None:
            snap[col] = None
        elif isinstance(val, bool):
            snap[col] = val
        elif isinstance(val, (np.bool_,)):
            snap[col] = bool(val)
        elif isinstance(val, dict):
            snap[col] = val
        elif isinstance(val, (float, np.floating)):
            snap[col] = None if np.isnan(val) else float(val)
        elif isinstance(val, (int, np.integer)):
            snap[col] = int(val)
        else:
            try:
                if pd.isna(val):
                    snap[col] = None
                else:
                    snap[col] = float(val)
            except (TypeError, ValueError):
                snap[col] = str(val)
    return snap


def classify_volatility(bar: pd.Series) -> str:
    atr_pct = bar.get("atr_pct_rank")
    if atr_pct is None or (isinstance(atr_pct, float) and math.isnan(atr_pct)):
        return "medium"
    atr_pct = float(atr_pct)
    if atr_pct < 0.33:
        return "low"
    if atr_pct < 0.66:
        return "medium"
    return "high"


def classify_funding(bar: pd.Series) -> str:
    rate = bar.get("funding_8h")
    if rate is None or (isinstance(rate, float) and math.isnan(rate)):
        return "neutral"
    rate = float(rate)
    if rate < -0.0001:
        return "negative"
    if rate > 0.0001:
        return "positive"
    return "neutral"


def classify_time_of_day(ts: datetime) -> str:
    hour = ts.hour
    if hour < 8:
        return "asia"
    if hour < 16:
        return "london"
    if hour < 22:
        return "new_york"
    return "off_hours"


# ---------------------------------------------------------------------------
# Strategy protocol
# ---------------------------------------------------------------------------

class Strategy(Protocol):
    name: str

    def on_bar(
        self, bar: pd.Series, prev_bar: pd.Series | None,
    ) -> EntryOrder | None: ...


# ---------------------------------------------------------------------------
# BB Squeeze strategy
# ---------------------------------------------------------------------------

class BBSqueezeStrategy:
    """
    Entry:  BB squeeze releases (True -> False) with MACD histogram
            confirming direction.
    SL:     2 x ATR from entry.
    TP:     3 x ATR from entry (1.5:1 R:R).
    """

    name = "bb_squeeze"

    def on_bar(
        self, bar: pd.Series, prev_bar: pd.Series | None,
    ) -> EntryOrder | None:
        if prev_bar is None:
            return None

        prev_squeeze = prev_bar.get("bb_squeeze")
        curr_squeeze = bar.get("bb_squeeze")

        if not (prev_squeeze == True and curr_squeeze == False):  # noqa: E712
            return None

        atr = bar.get("atr_14")
        macd_hist = bar.get("macd_hist")

        if atr is None or pd.isna(atr) or macd_hist is None or pd.isna(macd_hist):
            return None

        atr = float(atr)
        if atr <= 0:
            return None

        if float(macd_hist) > 0:
            direction = "long"
        else:
            direction = "short"

        return EntryOrder(
            direction=direction,
            sl_distance=2.0 * atr,
            tp_distance=3.0 * atr,
            strategy_combo=["bb_squeeze", "macd_confirm"],
            indicators_snapshot=build_indicator_snapshot(bar),
        )


STRATEGIES: dict[str, type[Strategy]] = {
    "bb_squeeze": BBSqueezeStrategy,
}


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class Simulator:

    def __init__(
        self,
        strategy: Strategy,
        symbol: str,
        leverage: int = 10,
        risk_pct: float = 0.02,
        equity: float = 10_000.0,
    ):
        self.strategy = strategy
        self.symbol = symbol
        self.leverage = leverage
        self.risk_pct = risk_pct
        self.equity = equity

    # ----- public -----------------------------------------------------------

    def run(
        self,
        engine,
        from_date: str,
        to_date: str,
    ) -> list[ClosedTrade]:
        bars = self._load_bars(engine, from_date, to_date)
        funding_map = self._load_funding(engine)

        log.info(
            "simulation_start",
            symbol=self.symbol,
            bars=len(bars),
            from_date=from_date,
            to_date=to_date,
        )

        position: _OpenPosition | None = None
        pending: EntryOrder | None = None
        closed: list[ClosedTrade] = []

        for i in range(len(bars)):
            bar = bars.iloc[i]
            prev_bar = bars.iloc[i - 1] if i > 0 else None
            bar_ts = pd.Timestamp(bar["timestamp"])

            # 1. Fill pending entry at this bar's open
            if pending is not None and position is None:
                position = self._fill_entry(pending, bar, i)
                pending = None

            # 2. Check exit (only for positions opened on a previous bar)
            elif position is not None and position.entry_bar_idx < i:
                trade = self._check_exit(position, bar)
                if trade is not None:
                    closed.append(trade)
                    position = None

            # 3. Apply funding if this bar lands on a funding timestamp
            if position is not None:
                self._apply_funding(position, bar_ts, funding_map)

            # 4. Generate new entry signal
            if position is None and pending is None:
                order = self.strategy.on_bar(bar, prev_bar)
                if order is not None:
                    pending = order

        # Force-close any remaining open position at last bar's close
        if position is not None:
            trade = self._force_close(position, bars.iloc[-1])
            closed.append(trade)

        log.info("simulation_complete", trades=len(closed))
        return closed

    # ----- data loading -----------------------------------------------------

    def _load_bars(self, engine, from_date: str, to_date: str) -> pd.DataFrame:
        candles = pd.read_sql(
            "SELECT * FROM candles_5m "
            "WHERE symbol = %(symbol)s "
            "  AND timestamp >= %(from_date)s "
            "  AND timestamp < %(to_date)s "
            "ORDER BY timestamp",
            engine,
            params={"symbol": self.symbol, "from_date": from_date, "to_date": to_date},
            parse_dates=["timestamp"],
        )
        indicators = pd.read_sql(
            "SELECT * FROM indicators_5m "
            "WHERE symbol = %(symbol)s "
            "  AND timestamp >= %(from_date)s "
            "  AND timestamp < %(to_date)s "
            "ORDER BY timestamp",
            engine,
            params={"symbol": self.symbol, "from_date": from_date, "to_date": to_date},
            parse_dates=["timestamp"],
        )
        merged = candles.merge(
            indicators,
            on=["symbol", "timestamp"],
            how="inner",
            suffixes=("", "_ind"),
        )
        log.info("bars_loaded", rows=len(merged))
        return merged

    def _load_funding(self, engine) -> dict[pd.Timestamp, float]:
        df = pd.read_sql(
            "SELECT timestamp, funding_rate FROM funding_history "
            "WHERE symbol = %(symbol)s ORDER BY timestamp",
            engine,
            params={"symbol": self.symbol},
            parse_dates=["timestamp"],
        )
        return {pd.Timestamp(row["timestamp"]): float(row["funding_rate"])
                for _, row in df.iterrows()}

    # ----- fill / exit logic ------------------------------------------------

    def _fill_entry(
        self, order: EntryOrder, bar: pd.Series, bar_idx: int,
    ) -> _OpenPosition:
        entry_price = float(bar["open"])

        if order.direction == "long":
            sl = entry_price - order.sl_distance
            tp = entry_price + order.tp_distance
        else:
            sl = entry_price + order.sl_distance
            tp = entry_price - order.tp_distance

        sl_pct = order.sl_distance / entry_price
        risk_amount = self.equity * self.risk_pct
        position_size_usd = risk_amount / sl_pct if sl_pct > 0 else risk_amount

        bar_ts = pd.Timestamp(bar["timestamp"])

        return _OpenPosition(
            direction=order.direction,
            entry_time=bar_ts.to_pydatetime(),
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            position_size_usd=position_size_usd,
            strategy_combo=order.strategy_combo,
            indicators_snapshot=order.indicators_snapshot,
            regime_volatility=classify_volatility(
                pd.Series(order.indicators_snapshot)
            ),
            regime_funding=classify_funding(
                pd.Series(order.indicators_snapshot)
            ),
            regime_time_of_day=classify_time_of_day(bar_ts.to_pydatetime()),
            entry_bar_idx=bar_idx,
        )

    def _check_exit(
        self, pos: _OpenPosition, bar: pd.Series,
    ) -> ClosedTrade | None:
        low = float(bar["low"])
        high = float(bar["high"])
        bar_ts = pd.Timestamp(bar["timestamp"]).to_pydatetime()

        sl_hit = False
        tp_hit = False

        if pos.direction == "long":
            sl_hit = low <= pos.stop_loss
            tp_hit = high >= pos.take_profit
        else:
            sl_hit = high >= pos.stop_loss
            tp_hit = low <= pos.take_profit

        if not sl_hit and not tp_hit:
            return None

        # Conservative: if both trigger on same bar, assume SL hit
        if sl_hit:
            exit_price = pos.stop_loss
            exit_reason = "sl"
        else:
            exit_price = pos.take_profit
            exit_reason = "tp"

        return self._close_position(pos, exit_price, bar_ts, exit_reason)

    def _force_close(
        self, pos: _OpenPosition, bar: pd.Series,
    ) -> ClosedTrade:
        exit_price = float(bar["close"])
        bar_ts = pd.Timestamp(bar["timestamp"]).to_pydatetime()
        return self._close_position(pos, exit_price, bar_ts, "timeout")

    def _close_position(
        self,
        pos: _OpenPosition,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
    ) -> ClosedTrade:
        if pos.direction == "long":
            raw_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            raw_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl_usd = pos.position_size_usd * raw_pct - pos.funding_paid_usd
        pnl_pct = raw_pct

        return ClosedTrade(
            symbol=self.symbol,
            direction=pos.direction,
            leverage=self.leverage,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            stop_loss=pos.stop_loss,
            take_profit=pos.take_profit,
            position_size_usd=pos.position_size_usd,
            pnl_pct=pnl_pct,
            pnl_usd=pnl_usd,
            funding_paid_usd=pos.funding_paid_usd,
            win_loss=pnl_usd > 0,
            strategy_combo=pos.strategy_combo,
            indicators_snapshot=pos.indicators_snapshot,
            regime_volatility=pos.regime_volatility,
            regime_funding=pos.regime_funding,
            regime_time_of_day=pos.regime_time_of_day,
            exit_reason=exit_reason,
        )

    # ----- funding ----------------------------------------------------------

    def _apply_funding(
        self,
        pos: _OpenPosition,
        bar_ts: pd.Timestamp,
        funding_map: dict[pd.Timestamp, float],
    ) -> None:
        rate = funding_map.get(bar_ts)
        if rate is None:
            return
        if pos.direction == "long":
            pos.funding_paid_usd += pos.position_size_usd * rate
        else:
            pos.funding_paid_usd -= pos.position_size_usd * rate
