"""
Bar-by-bar backtest simulator with hybrid fill model.

- Limit + maker (0.02% / side): mean-reversion (VWAP, RSI divergence).
  Fill when price trades through limit_price; TTL = LIMIT_ORDER_TTL bars.
- Market + taker (0.055% / side): momentum / breakouts (BB squeeze,
  multitf scalp, volume delta). Fill at next bar's open after signal.
- Cooldown between trades reduces overtrading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd
import structlog
from sqlalchemy import create_engine

log = structlog.get_logger()

MAKER_FEE = 0.0002        # Bybit maker, per side
TAKER_FEE = 0.00055       # Bybit taker, per side (VIP0)
LIMIT_ORDER_TTL = 3       # bars before unfilled limit expires
COOLDOWN_BARS = 6         # minimum bars between trades (30 min on 5m)

INDICATOR_COLS = [
    "ema_9", "ema_21", "ema_50", "ema_200",
    "rsi_14", "mfi_14", "stochrsi_k", "stochrsi_d",
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
    limit_price: float = 0.0
    fill_mode: str = "limit"  # "limit" | "market"


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
    fees_paid_usd: float
    win_loss: bool
    strategy_combo: list[str]
    indicators_snapshot: dict
    regime_volatility: str
    regime_funding: str
    regime_time_of_day: str
    exit_reason: str
    mfe_pct: float = 0.0
    mae_pct: float = 0.0
    bars_held: int = 0
    notes: str | None = None


@dataclass
class _OpenPosition:
    direction: str
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    original_stop_loss: float
    position_size_usd: float
    strategy_combo: list[str]
    indicators_snapshot: dict
    regime_volatility: str
    regime_funding: str
    regime_time_of_day: str
    entry_bar_idx: int
    funding_paid_usd: float = 0.0
    fees_paid_usd: float = 0.0
    fee_rate_per_side: float = MAKER_FEE
    be_activated: bool = False
    mfe_pct: float = 0.0
    mae_pct: float = 0.0


@dataclass
class _PendingOrder:
    order: EntryOrder
    limit_price: float
    signal_bar_idx: int
    bars_alive: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_indicator_snapshot(bar: pd.Series) -> dict:
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
        fixed_risk: bool = False,
        breakeven_pct: float | None = None,
    ):
        self.strategy = strategy
        self.symbol = symbol
        self.leverage = leverage
        self.risk_pct = risk_pct
        self.equity = equity
        self._initial_equity = equity
        self.fixed_risk = fixed_risk
        self.breakeven_pct = breakeven_pct

    def run(
        self,
        engine,
        from_date: str,
        to_date: str,
        bars: pd.DataFrame | None = None,
    ) -> list[ClosedTrade]:
        if bars is not None:
            bar_data = bars
            log.info("bars_loaded", rows=len(bar_data), source="prebuilt")
        else:
            bar_data = self._load_bars(engine, from_date, to_date)
        funding_map = self._load_funding(engine)

        log.info(
            "simulation_start",
            symbol=self.symbol,
            bars=len(bar_data),
            from_date=from_date,
            to_date=to_date,
        )

        position: _OpenPosition | None = None
        pending: _PendingOrder | None = None
        closed: list[ClosedTrade] = []
        last_exit_bar: int = -COOLDOWN_BARS
        unfilled_count: int = 0

        for i in range(len(bar_data)):
            bar = bar_data.iloc[i]
            prev_bar = bar_data.iloc[i - 1] if i > 0 else None
            bar_ts = pd.Timestamp(bar["timestamp"])
            bar_open = float(bar["open"])
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            # 1. Fill pending order (limit touch or next-bar market open)
            if pending is not None and position is None:
                pending.bars_alive += 1
                filled = False
                fill_price = 0.0
                odr = pending.order
                is_market = odr.fill_mode == "market"

                if is_market:
                    if pending.bars_alive == 1:
                        fill_price = bar_open
                        filled = True
                    elif pending.bars_alive > 1:
                        unfilled_count += 1
                        pending = None
                else:
                    if odr.direction == "long":
                        if bar_low <= pending.limit_price:
                            fill_price = pending.limit_price
                            filled = True
                    else:
                        if bar_high >= pending.limit_price:
                            fill_price = pending.limit_price
                            filled = True
                    if not filled and pending.bars_alive >= LIMIT_ORDER_TTL:
                        unfilled_count += 1
                        pending = None

                if filled and pending is not None:
                    fee = TAKER_FEE if is_market else MAKER_FEE
                    position = self._fill_entry(
                        odr, fill_price, bar, i, fee,
                    )
                    same_bar_exit = self._check_same_bar_exit(
                        position, bar_high, bar_low,
                    )
                    if same_bar_exit is not None:
                        trade = self._close_position(
                            position, same_bar_exit,
                            bar_ts.to_pydatetime(), "sl",
                        )
                        closed.append(trade)
                        position = None
                        last_exit_bar = i
                    pending = None

            # 2. Check breakeven activation + exit for existing position
            if position is not None and position.entry_bar_idx < i:
                self._update_mfe_mae(position, bar_high, bar_low)
                if (self.breakeven_pct is not None
                        and not position.be_activated):
                    self._check_breakeven(position, bar_high, bar_low)
                trade = self._check_exit(position, bar)
                if trade is not None:
                    closed.append(trade)
                    position = None
                    last_exit_bar = i

            # 3. Apply funding
            if position is not None:
                self._apply_funding(position, bar_ts, funding_map)

            # 4. Generate new signal (with cooldown)
            if (position is None and pending is None
                    and (i - last_exit_bar) >= COOLDOWN_BARS):
                order = self.strategy.on_bar(bar, prev_bar)
                if order is not None:
                    limit_px = (
                        order.limit_price if order.limit_price > 0 else bar_close
                    )
                    pending = _PendingOrder(
                        order=order,
                        limit_price=limit_px,
                        signal_bar_idx=i,
                    )

        if pending is not None:
            unfilled_count += 1

        if position is not None:
            trade = self._force_close(position, bar_data.iloc[-1])
            closed.append(trade)

        log.info(
            "simulation_complete",
            trades=len(closed),
            unfilled=unfilled_count,
        )
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
            params={"symbol": self.symbol, "from_date": from_date,
                    "to_date": to_date},
            parse_dates=["timestamp"],
        )
        indicators = pd.read_sql(
            "SELECT * FROM indicators_5m "
            "WHERE symbol = %(symbol)s "
            "  AND timestamp >= %(from_date)s "
            "  AND timestamp < %(to_date)s "
            "ORDER BY timestamp",
            engine,
            params={"symbol": self.symbol, "from_date": from_date,
                    "to_date": to_date},
            parse_dates=["timestamp"],
        )
        merged = candles.merge(
            indicators,
            on=["symbol", "timestamp"],
            how="inner",
            suffixes=("", "_ind"),
        )
        numeric_cols = ["open", "high", "low", "close", "volume",
                        "buy_volume", "sell_volume"]
        for col in numeric_cols:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")
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

    # ----- fill logic -------------------------------------------------------

    def _fill_entry(
        self,
        order: EntryOrder,
        fill_price: float,
        bar: pd.Series,
        bar_idx: int,
        fee_rate: float,
    ) -> _OpenPosition:
        if order.direction == "long":
            sl = fill_price - order.sl_distance
            tp = fill_price + order.tp_distance
        else:
            sl = fill_price + order.sl_distance
            tp = fill_price - order.tp_distance

        sl_pct = order.sl_distance / fill_price if fill_price > 0 else 0.01
        base_equity = self._initial_equity if self.fixed_risk else self.equity
        risk_amount = base_equity * self.risk_pct
        position_size_usd = risk_amount / sl_pct if sl_pct > 0 else risk_amount
        position_size_usd = min(position_size_usd, self.equity * self.leverage)

        entry_fee = position_size_usd * fee_rate
        bar_ts = pd.Timestamp(bar["timestamp"])

        return _OpenPosition(
            direction=order.direction,
            entry_time=bar_ts.to_pydatetime(),
            entry_price=fill_price,
            stop_loss=sl,
            take_profit=tp,
            original_stop_loss=sl,
            position_size_usd=position_size_usd,
            strategy_combo=order.strategy_combo,
            indicators_snapshot=order.indicators_snapshot,
            regime_volatility=classify_volatility(
                pd.Series(order.indicators_snapshot)),
            regime_funding=classify_funding(
                pd.Series(order.indicators_snapshot)),
            regime_time_of_day=classify_time_of_day(bar_ts.to_pydatetime()),
            entry_bar_idx=bar_idx,
            fees_paid_usd=entry_fee,
            fee_rate_per_side=fee_rate,
        )

    def _update_mfe_mae(
        self, pos: _OpenPosition, bar_high: float, bar_low: float,
    ) -> None:
        entry = pos.entry_price
        if entry <= 0:
            return
        if pos.direction == "long":
            fav = (bar_high - entry) / entry
            adv = (entry - bar_low) / entry
        else:
            fav = (entry - bar_low) / entry
            adv = (bar_high - entry) / entry
        if fav > pos.mfe_pct:
            pos.mfe_pct = fav
        if adv > pos.mae_pct:
            pos.mae_pct = adv

    def _check_breakeven(
        self, pos: _OpenPosition, bar_high: float, bar_low: float,
    ) -> None:
        """Move SL to entry (breakeven) if unrealized gain hits threshold."""
        threshold = self.breakeven_pct
        if threshold is None:
            return
        entry = pos.entry_price
        if pos.direction == "long":
            target = entry * (1 + threshold)
            if bar_high >= target:
                pos.stop_loss = entry
                pos.be_activated = True
        else:
            target = entry * (1 - threshold)
            if bar_low <= target:
                pos.stop_loss = entry
                pos.be_activated = True

    def _check_same_bar_exit(
        self, pos: _OpenPosition, bar_high: float, bar_low: float,
    ) -> float | None:
        """Return SL price if stop is hit on the fill bar."""
        if pos.direction == "long" and bar_low <= pos.stop_loss:
            return pos.stop_loss
        if pos.direction == "short" and bar_high >= pos.stop_loss:
            return pos.stop_loss
        return None

    def _check_exit(
        self, pos: _OpenPosition, bar: pd.Series,
    ) -> ClosedTrade | None:
        low = float(bar["low"])
        high = float(bar["high"])
        bar_ts = pd.Timestamp(bar["timestamp"]).to_pydatetime()

        if pos.direction == "long":
            sl_hit = low <= pos.stop_loss
            tp_hit = high >= pos.take_profit
        else:
            sl_hit = high >= pos.stop_loss
            tp_hit = low <= pos.take_profit

        if not sl_hit and not tp_hit:
            return None

        if sl_hit and tp_hit:
            exit_price = pos.stop_loss
            exit_reason = "be" if pos.be_activated else "sl"
        elif sl_hit:
            exit_price = pos.stop_loss
            exit_reason = "be" if pos.be_activated else "sl"
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

        exit_fee = pos.position_size_usd * pos.fee_rate_per_side
        total_fees = pos.fees_paid_usd + exit_fee

        pnl_usd = (pos.position_size_usd * raw_pct
                    - pos.funding_paid_usd - total_fees)
        pnl_pct = pnl_usd / pos.position_size_usd if pos.position_size_usd > 0 else 0.0

        self.equity += pnl_usd

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
            fees_paid_usd=total_fees,
            win_loss=pnl_usd > 0,
            strategy_combo=pos.strategy_combo,
            indicators_snapshot=pos.indicators_snapshot,
            regime_volatility=pos.regime_volatility,
            regime_funding=pos.regime_funding,
            regime_time_of_day=pos.regime_time_of_day,
            exit_reason=exit_reason,
            mfe_pct=pos.mfe_pct,
            mae_pct=pos.mae_pct,
        )

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
