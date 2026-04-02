import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.db.base import Base


class Candles5m(Base):
    __tablename__ = "candles_5m"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    open: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[float] = mapped_column(Numeric(30, 8), nullable=False)
    buy_volume: Mapped[float | None] = mapped_column(Numeric(30, 8))
    sell_volume: Mapped[float | None] = mapped_column(Numeric(30, 8))
    volume_delta: Mapped[float | None] = mapped_column(Numeric(30, 8))
    quote_volume: Mapped[float | None] = mapped_column(Numeric(30, 8))
    mark_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    funding_rate: Mapped[float | None] = mapped_column(Numeric(20, 10))
    trade_count: Mapped[int | None] = mapped_column(Integer)


class Candles1m(Base):
    __tablename__ = "candles_1m"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    open: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    high: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    low: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    close: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    volume: Mapped[float] = mapped_column(Numeric(30, 8), nullable=False)
    buy_volume: Mapped[float | None] = mapped_column(Numeric(30, 8))
    sell_volume: Mapped[float | None] = mapped_column(Numeric(30, 8))
    volume_delta: Mapped[float | None] = mapped_column(Numeric(30, 8))
    quote_volume: Mapped[float | None] = mapped_column(Numeric(30, 8))
    mark_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    funding_rate: Mapped[float | None] = mapped_column(Numeric(20, 10))
    trade_count: Mapped[int | None] = mapped_column(Integer)


class RawTrades(Base):
    __tablename__ = "raw_trades"

    id: Mapped[int] = mapped_column(
        Integer, autoincrement=True, nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    price: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    qty: Mapped[float] = mapped_column(Numeric(30, 8), nullable=False)
    side: Mapped[str] = mapped_column(String(4), nullable=False)
    trade_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    is_liquidation: Mapped[bool | None] = mapped_column(
        Boolean, server_default=text("false")
    )


class FundingHistory(Base):
    __tablename__ = "funding_history"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    funding_rate: Mapped[float] = mapped_column(Numeric(20, 10), nullable=False)
    mark_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    index_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    predicted_rate: Mapped[float | None] = mapped_column(Numeric(20, 10))


class Liquidations(Base):
    __tablename__ = "liquidations"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    side: Mapped[str] = mapped_column(String(5), primary_key=True)
    qty: Mapped[float] = mapped_column(Numeric(30, 8), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(20, 8), primary_key=True)
    value_usd: Mapped[float | None] = mapped_column(Numeric(30, 8))


class Indicators5m(Base):
    __tablename__ = "indicators_5m"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )

    # Trend
    ema_9: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ema_21: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ema_50: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ema_200: Mapped[float | None] = mapped_column(Numeric(20, 8))

    # Momentum
    rsi_14: Mapped[float | None] = mapped_column(Numeric(10, 4))
    mfi_14: Mapped[float | None] = mapped_column(Numeric(10, 4))
    stochrsi_k: Mapped[float | None] = mapped_column(Numeric(10, 4))
    stochrsi_d: Mapped[float | None] = mapped_column(Numeric(10, 4))
    macd_line: Mapped[float | None] = mapped_column(Numeric(20, 8))
    macd_signal: Mapped[float | None] = mapped_column(Numeric(20, 8))
    macd_hist: Mapped[float | None] = mapped_column(Numeric(20, 8))
    macd_fast_line: Mapped[float | None] = mapped_column(Numeric(20, 8))
    macd_fast_signal: Mapped[float | None] = mapped_column(Numeric(20, 8))
    macd_fast_hist: Mapped[float | None] = mapped_column(Numeric(20, 8))

    # Volatility / bands
    bb_upper: Mapped[float | None] = mapped_column(Numeric(20, 8))
    bb_mid: Mapped[float | None] = mapped_column(Numeric(20, 8))
    bb_lower: Mapped[float | None] = mapped_column(Numeric(20, 8))
    bb_width: Mapped[float | None] = mapped_column(Numeric(20, 8))
    kc_upper: Mapped[float | None] = mapped_column(Numeric(20, 8))
    kc_lower: Mapped[float | None] = mapped_column(Numeric(20, 8))
    bb_squeeze: Mapped[bool | None] = mapped_column(Boolean)
    atr_14: Mapped[float | None] = mapped_column(Numeric(20, 8))
    atr_pct_rank: Mapped[float | None] = mapped_column(Numeric(10, 4))

    # Structure
    supertrend: Mapped[float | None] = mapped_column(Numeric(20, 8))
    supertrend_dir: Mapped[int | None] = mapped_column(SmallInteger)
    vwap: Mapped[float | None] = mapped_column(Numeric(20, 8))
    vwap_dev_upper1: Mapped[float | None] = mapped_column(Numeric(20, 8))
    vwap_dev_lower1: Mapped[float | None] = mapped_column(Numeric(20, 8))
    vwap_dev_upper2: Mapped[float | None] = mapped_column(Numeric(20, 8))
    vwap_dev_lower2: Mapped[float | None] = mapped_column(Numeric(20, 8))

    # Volume / flow
    obv: Mapped[float | None] = mapped_column(Numeric(30, 8))
    volume_delta: Mapped[float | None] = mapped_column(Numeric(30, 8))
    order_flow_imb: Mapped[float | None] = mapped_column(Numeric(10, 6))

    # Heikin Ashi
    ha_open: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ha_high: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ha_low: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ha_close: Mapped[float | None] = mapped_column(Numeric(20, 8))

    # Regime
    funding_8h: Mapped[float | None] = mapped_column(Numeric(20, 10))
    funding_24h_cum: Mapped[float | None] = mapped_column(Numeric(20, 10))
    liq_volume_1h: Mapped[float | None] = mapped_column(Numeric(30, 8))

    # Overflow
    extras: Mapped[dict | None] = mapped_column(JSONB)


class Indicators15m(Base):
    __tablename__ = "indicators_15m"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )

    # Trend
    ema_9: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ema_21: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ema_50: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ema_200: Mapped[float | None] = mapped_column(Numeric(20, 8))

    # Momentum
    rsi_14: Mapped[float | None] = mapped_column(Numeric(10, 4))
    mfi_14: Mapped[float | None] = mapped_column(Numeric(10, 4))
    stochrsi_k: Mapped[float | None] = mapped_column(Numeric(10, 4))
    stochrsi_d: Mapped[float | None] = mapped_column(Numeric(10, 4))
    macd_line: Mapped[float | None] = mapped_column(Numeric(20, 8))
    macd_signal: Mapped[float | None] = mapped_column(Numeric(20, 8))
    macd_hist: Mapped[float | None] = mapped_column(Numeric(20, 8))
    macd_fast_line: Mapped[float | None] = mapped_column(Numeric(20, 8))
    macd_fast_signal: Mapped[float | None] = mapped_column(Numeric(20, 8))
    macd_fast_hist: Mapped[float | None] = mapped_column(Numeric(20, 8))

    # Volatility / bands
    bb_upper: Mapped[float | None] = mapped_column(Numeric(20, 8))
    bb_mid: Mapped[float | None] = mapped_column(Numeric(20, 8))
    bb_lower: Mapped[float | None] = mapped_column(Numeric(20, 8))
    bb_width: Mapped[float | None] = mapped_column(Numeric(20, 8))
    kc_upper: Mapped[float | None] = mapped_column(Numeric(20, 8))
    kc_lower: Mapped[float | None] = mapped_column(Numeric(20, 8))
    bb_squeeze: Mapped[bool | None] = mapped_column(Boolean)
    atr_14: Mapped[float | None] = mapped_column(Numeric(20, 8))
    atr_pct_rank: Mapped[float | None] = mapped_column(Numeric(10, 4))

    # Structure
    supertrend: Mapped[float | None] = mapped_column(Numeric(20, 8))
    supertrend_dir: Mapped[int | None] = mapped_column(SmallInteger)
    vwap: Mapped[float | None] = mapped_column(Numeric(20, 8))
    vwap_dev_upper1: Mapped[float | None] = mapped_column(Numeric(20, 8))
    vwap_dev_lower1: Mapped[float | None] = mapped_column(Numeric(20, 8))
    vwap_dev_upper2: Mapped[float | None] = mapped_column(Numeric(20, 8))
    vwap_dev_lower2: Mapped[float | None] = mapped_column(Numeric(20, 8))

    # Volume / flow
    obv: Mapped[float | None] = mapped_column(Numeric(30, 8))
    volume_delta: Mapped[float | None] = mapped_column(Numeric(30, 8))
    order_flow_imb: Mapped[float | None] = mapped_column(Numeric(10, 6))

    # Heikin Ashi
    ha_open: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ha_high: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ha_low: Mapped[float | None] = mapped_column(Numeric(20, 8))
    ha_close: Mapped[float | None] = mapped_column(Numeric(20, 8))

    # Regime
    funding_8h: Mapped[float | None] = mapped_column(Numeric(20, 10))
    funding_24h_cum: Mapped[float | None] = mapped_column(Numeric(20, 10))
    liq_volume_1h: Mapped[float | None] = mapped_column(Numeric(30, 8))

    # Overflow
    extras: Mapped[dict | None] = mapped_column(JSONB)


class TradesLog(Base):
    __tablename__ = "trades_log"

    trade_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    direction: Mapped[str] = mapped_column(String(5), nullable=False)
    leverage: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    entry_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    exit_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    entry_price: Mapped[float] = mapped_column(Numeric(20, 8), nullable=False)
    exit_price: Mapped[float | None] = mapped_column(Numeric(20, 8))
    stop_loss: Mapped[float | None] = mapped_column(Numeric(20, 8))
    take_profit: Mapped[float | None] = mapped_column(Numeric(20, 8))
    position_size_usd: Mapped[float | None] = mapped_column(Numeric(20, 4))
    pnl_pct: Mapped[float | None] = mapped_column(Numeric(10, 6))
    pnl_usd: Mapped[float | None] = mapped_column(Numeric(20, 4))
    funding_paid_usd: Mapped[float | None] = mapped_column(
        Numeric(20, 4), server_default=text("0")
    )
    win_loss: Mapped[bool | None] = mapped_column(Boolean)
    strategy_combo: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False
    )
    indicators_snapshot: Mapped[dict] = mapped_column(JSONB, nullable=False)
    regime_volatility: Mapped[str | None] = mapped_column(String(10))
    regime_funding: Mapped[str | None] = mapped_column(String(10))
    regime_time_of_day: Mapped[str | None] = mapped_column(String(15))
    exit_reason: Mapped[str | None] = mapped_column(String(20))
    is_backtest: Mapped[bool | None] = mapped_column(
        Boolean, server_default=text("true")
    )
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )


class CoinglassLiquidationBar(Base):
    """Aggregated long/short liquidation USD from CoinGlass API (e.g. 4h bars)."""

    __tablename__ = "coinglass_liquidation_bars"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    bucket_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), primary_key=True
    )
    interval: Mapped[str] = mapped_column(String(10), primary_key=True)
    long_liquidation_usd: Mapped[float] = mapped_column(
        Numeric(30, 4), nullable=False
    )
    short_liquidation_usd: Mapped[float] = mapped_column(
        Numeric(30, 4), nullable=False
    )
    fetched_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )


class StrategyPerformance(Base):
    __tablename__ = "strategy_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    strategy_combo: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(5), nullable=False)
    total_trades: Mapped[int | None] = mapped_column(Integer)
    win_rate: Mapped[float | None] = mapped_column(Numeric(6, 4))
    avg_pnl_pct: Mapped[float | None] = mapped_column(Numeric(10, 6))
    expectancy: Mapped[float | None] = mapped_column(Numeric(10, 6))
    sharpe: Mapped[float | None] = mapped_column(Numeric(10, 4))
    max_drawdown: Mapped[float | None] = mapped_column(Numeric(10, 6))
    best_regime: Mapped[str | None] = mapped_column(String(20))
    worst_regime: Mapped[str | None] = mapped_column(String(20))
    regime_funding: Mapped[str | None] = mapped_column(String(10))
    last_updated: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=text("now()")
    )

    __table_args__ = (
        UniqueConstraint(
            "strategy_combo", "symbol", "timeframe", "regime_funding",
            name="uq_strategy_performance_combo",
        ),
    )
