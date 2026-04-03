"""
Connects to Bybit WebSocket and maintains a rolling window of the last
200 5m bars in memory for each symbol.

On each closed 5m bar
  1. Writes the new candle to candles_5m
  2. Computes indicators for the new bar (recompute on the 200-bar buffer)
  3. Writes indicators to indicators_5m (and indicators_15m when applicable)
  4. Calls each active strategy's generate_signal()
  5. Passes any non-None signal to RiskManager.evaluate()
  6. Passes approved signals to OrderManager.execute()

Usage
-----
    python src/live/websocket_listener.py
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import pandas_ta as ta  # noqa: F401 — registers .ta accessor
except ModuleNotFoundError:
    import pandas_ta_classic as ta  # noqa: F401 — drop-in fork
import structlog
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import settings  # noqa: E402
from src.db.models import Candles5m, Indicators5m, Indicators15m  # noqa: E402
from src.indicators.compute_all import (  # noqa: E402
    TA_COL_MAP,
    DERIVED_FLOAT_COLS,
    compute_derived,
    compute_ta_indicators,
    pack_indicator_extras,
)
from src.indicators.custom_indicators import (  # noqa: E402
    detect_momentum_divergence,
    detect_rsi_divergence,
)
from src.indicators.resample import (  # noqa: E402
    resample_5m_to_15m,
    resample_candles,
    CONTEXT_TF,
)
from src.live.order_manager import OrderManager  # noqa: E402
from src.live.telegram_notifier import TelegramNotifier  # noqa: E402
from src.risk.risk_manager import RiskManager  # noqa: E402
from src.strategies import STRATEGY_REGISTRY  # noqa: E402
from src.strategies.base import BaseStrategy, _sf  # noqa: E402
from src.strategies.strategy_sniper import SniperStrategy  # noqa: E402

log = structlog.get_logger()

# Minimum buffer length before we start running indicators / strategies
_MIN_BUFFER = 20
# Bars to keep in memory per symbol (5000 = ~17 days of 5m bars, enough for 4h indicators)
_BUFFER_SIZE = 5000


# -----------------------------------------------------------------------
# Shared bot state — read by the health API, written by the listener
# -----------------------------------------------------------------------

class BotState:
    """Thread-safe shared state for the health API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running: bool = False
        self.start_time: datetime | None = None
        self.symbols: list[str] = []
        self.ws_connected: bool = False
        self.last_bar: dict[str, str] = {}
        self.blocked_signals: list[dict] = []

    def update_last_bar(self, symbol: str, ts: datetime) -> None:
        with self._lock:
            self.last_bar[symbol] = ts.isoformat()

    def add_blocked(self, info: dict) -> None:
        with self._lock:
            self.blocked_signals.append(info)
            if len(self.blocked_signals) > 200:
                self.blocked_signals = self.blocked_signals[-200:]


# -----------------------------------------------------------------------
# WebSocket Listener
# -----------------------------------------------------------------------

class WebSocketListener:
    """
    Core live-trading engine.  Subscribes to Bybit WebSocket streams,
    processes closed 5m bars, and drives the strategy → risk → order
    pipeline.
    """

    def __init__(self, engine: Any, state: BotState) -> None:
        self._engine = engine
        self.state = state

        self._bar_buffer: dict[str, deque] = {}
        self._db_unavailable: bool = False
        self._latest_funding: dict[str, float] = {}
        self._predicted_funding: dict[str, float] = {}
        self._latest_mark: dict[str, float] = {}
        self._latest_15m: dict[str, pd.Series] = {}

        wanted = settings.live_strategy.lower()

        self._strategies: list[BaseStrategy] = []
        for name, cls in STRATEGY_REGISTRY.items():
            if wanted != "all" and name != wanted:
                continue
            try:
                if name == "regime_adaptive":
                    self._strategies.append(cls(engine=engine))
                else:
                    self._strategies.append(cls())
            except Exception as exc:
                log.warning("strategy_init_failed",
                            strategy=name, error=str(exc))

        self._sniper_15m: dict[str, SniperStrategy] = {}
        self._sniper_4h: dict[str, SniperStrategy] = {}
        self._sniper_tfs: list[str] = ["15m", "4h"]
        self._last_signals: dict[str, dict] = {}

        self.risk_manager = RiskManager(
            is_backtest=False, risk_pct=settings.live_risk_pct,
        )
        self.order_manager = OrderManager(engine)
        self._equity = settings.live_equity
        self.telegram = TelegramNotifier()

        log.info("listener_initialized",
                 strategies=[s.name for s in self._strategies],
                 sniper_tfs=self._sniper_tfs,
                 symbols=settings.symbols,
                 equity=self._equity,
                 risk_pct=settings.live_risk_pct)

    # ----- public -------------------------------------------------------

    def start(self) -> None:
        """Connect WebSocket streams and begin processing."""
        self.state.running = True
        self.state.start_time = datetime.now(timezone.utc)
        self.state.symbols = list(settings.symbols)

        for symbol in settings.symbols:
            self._bar_buffer[symbol] = deque(maxlen=_BUFFER_SIZE)
            self._init_buffer(symbol)

        self._connect_ws()

        self.telegram.notify_startup(
            symbols=settings.symbols,
            equity=self._equity,
            risk_pct=settings.live_risk_pct,
            strategy=settings.live_strategy,
        )
        exchange = self.order_manager._exchange
        self.telegram.start_hourly_loop(exchange=exchange)

    # ----- initialisation -----------------------------------------------

    def _init_buffer(self, symbol: str) -> None:
        """Pre-fill bar buffer from DB, falling back to Bybit REST API."""
        loaded = False

        # Try DB first
        try:
            df = pd.read_sql(
                "SELECT * FROM candles_5m WHERE symbol = %(symbol)s "
                "ORDER BY timestamp DESC LIMIT %(n)s",
                self._engine,
                params={"symbol": symbol, "n": _BUFFER_SIZE},
                parse_dates=["timestamp"],
            )
            if not df.empty:
                for _, row in df.sort_values("timestamp").iterrows():
                    self._bar_buffer[symbol].append(row.to_dict())
                log.info("buffer_from_db",
                         symbol=symbol, bars=len(df))
                loaded = True
        except Exception:
            pass

        # Fall back to Bybit REST API
        if not loaded:
            try:
                self._init_buffer_from_bybit(symbol)
            except Exception as exc:
                log.warning("buffer_init_failed",
                            symbol=symbol, error=str(exc))

    def _init_buffer_from_bybit(self, symbol: str) -> None:
        """Download recent 5m candles from Bybit REST API into the buffer."""
        from src.live.order_manager import _to_ccxt_symbol
        import ccxt

        exchange = ccxt.bybit({"enableRateLimit": True})
        exchange.load_markets()
        ccxt_sym = _to_ccxt_symbol(symbol)

        if ccxt_sym not in exchange.markets:
            log.warning("buffer_symbol_not_found", symbol=symbol)
            return

        now_ms = int(time.time() * 1000)
        target_bars = _BUFFER_SIZE
        five_min_ms = 5 * 60 * 1000
        since = now_ms - (target_bars * five_min_ms)

        all_candles = []
        current = since
        limit = 1000

        while current < now_ms and len(all_candles) < target_bars:
            ohlcv = exchange.fetch_ohlcv(
                ccxt_sym, "5m", since=current, limit=limit,
            )
            if not ohlcv:
                break
            for c in ohlcv:
                if c[0] < now_ms:
                    all_candles.append(c)
            last_ts = ohlcv[-1][0]
            if last_ts <= current:
                break
            current = last_ts + 1

        for c in all_candles:
            bar = {
                "symbol": symbol,
                "timestamp": datetime.fromtimestamp(
                    c[0] / 1000, tz=timezone.utc,
                ),
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": float(c[5]),
                "buy_volume": None,
                "sell_volume": None,
                "volume_delta": None,
                "quote_volume": float(c[5]) * float(c[4]),
                "mark_price": None,
                "funding_rate": None,
                "trade_count": None,
            }
            self._bar_buffer[symbol].append(bar)

        log.info("buffer_from_bybit",
                 symbol=symbol, bars=len(all_candles))

    def _connect_ws(self) -> None:
        """Connect public and (optionally) private WebSocket streams."""
        from pybit.unified_trading import WebSocket

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                ws_kwargs = {
                    "testnet": settings.bybit_testnet,
                    "channel_type": "linear",
                }
                if settings.bybit_demo:
                    ws_kwargs["testnet"] = False
                    ws_kwargs["demo"] = True

                self._ws_public = WebSocket(**ws_kwargs)
                for symbol in settings.symbols:
                    self._ws_public.kline_stream(
                        interval=5,
                        symbol=symbol,
                        callback=self._on_kline,
                    )
                    self._ws_public.ticker_stream(
                        symbol=symbol,
                        callback=self._on_ticker,
                    )
                self.state.ws_connected = True
                log.info("ws_public_connected",
                         symbols=settings.symbols, attempt=attempt)
                break
            except Exception as exc:
                wait = min(2 ** attempt, 60)
                log.warning("ws_connect_retry",
                            attempt=attempt, wait=wait, error=str(exc))
                time.sleep(wait)
        else:
            log.error("ws_connect_failed_all_retries")
            return

        if settings.bybit_api_key and settings.bybit_api_secret:
            try:
                priv_kwargs = {
                    "testnet": settings.bybit_testnet,
                    "channel_type": "private",
                    "api_key": settings.bybit_api_key,
                    "api_secret": settings.bybit_api_secret,
                }
                if settings.bybit_demo:
                    priv_kwargs["testnet"] = False
                    priv_kwargs["demo"] = True

                self._ws_private = WebSocket(**priv_kwargs)
                self._ws_private.execution_stream(
                    callback=self._on_execution,
                )
                log.info("ws_private_connected")
            except Exception as exc:
                log.warning("ws_private_failed", error=str(exc))
        else:
            log.warning("ws_private_skipped",
                        reason="API keys not configured")

    # ----- WebSocket callbacks ------------------------------------------

    def _on_kline(self, message: dict) -> None:
        try:
            topic = message.get("topic", "")
            parts = topic.split(".")
            symbol = parts[2] if len(parts) >= 3 else ""

            for bar_data in message.get("data", []):
                if not bar_data.get("confirm"):
                    continue

                bar = {
                    "symbol": symbol,
                    "timestamp": datetime.fromtimestamp(
                        bar_data["start"] / 1000, tz=timezone.utc,
                    ),
                    "open": float(bar_data["open"]),
                    "high": float(bar_data["high"]),
                    "low": float(bar_data["low"]),
                    "close": float(bar_data["close"]),
                    "volume": float(bar_data["volume"]),
                    "buy_volume": None,
                    "sell_volume": None,
                    "volume_delta": None,
                    "quote_volume": float(
                        bar_data.get("turnover", 0) or 0
                    ),
                    "mark_price": self._latest_mark.get(symbol),
                    "funding_rate": None,
                    "trade_count": None,
                }

                self._process_closed_bar(symbol, bar)
        except Exception as exc:
            log.error("kline_handler_error", error=str(exc))

    def _on_ticker(self, message: dict) -> None:
        try:
            data = message.get("data", {})
            symbol = data.get("symbol", "")
            fr = data.get("fundingRate")
            if fr:
                self._latest_funding[symbol] = float(fr)
            mp = data.get("markPrice")
            if mp:
                self._latest_mark[symbol] = float(mp)
        except Exception as exc:
            log.error("ticker_handler_error", error=str(exc))

    def _on_execution(self, message: dict) -> None:
        try:
            for fill in message.get("data", []):
                symbol = fill.get("symbol", "")
                side = fill.get("side", "")
                order_id = fill.get("orderId", "")

                tracked = self.order_manager._open_orders.get(symbol, {})
                direction = tracked.get("direction", "")
                entry_price = tracked.get("price", 0)

                is_close = bool(tracked) and (
                    (direction == "long" and side.lower() == "sell")
                    or (direction == "short" and side.lower() == "buy")
                )

                signal_info = self._last_signals.get(symbol, {})
                sl = signal_info.get("sl", 0)
                tp = signal_info.get("tp", 0)

                self.order_manager.handle_fill(fill)
                self.telegram.notify_fill(
                    symbol=symbol,
                    side=side,
                    price=float(fill.get("execPrice", 0) or 0),
                    qty=float(fill.get("execQty", 0) or 0),
                    order_id=order_id,
                    sl=sl, tp=tp,
                    is_close=is_close,
                    entry_price=entry_price,
                    direction=direction,
                )
        except Exception as exc:
            log.error("execution_handler_error", error=str(exc))

    # ----- bar processing pipeline --------------------------------------

    def _process_closed_bar(self, symbol: str, bar: dict) -> None:
        """End-to-end processing of a single confirmed 5m bar."""
        try:
            self._bar_buffer[symbol].append(bar)
            self._write_candle(bar)

            buf_len = len(self._bar_buffer[symbol])
            if buf_len < _MIN_BUFFER:
                log.info("buffer_warming",
                         symbol=symbol, bars=buf_len,
                         need=_MIN_BUFFER)
                return

            raw_5m, ind_5m, raw_15m, ind_15m = (
                self._compute_indicators(symbol)
            )
            if raw_5m is None:
                log.warning("indicators_none", symbol=symbol)
                return

            self._write_indicator_row(symbol, raw_5m, Indicators5m)
            if raw_15m is not None:
                self._write_indicator_row(symbol, raw_15m, Indicators15m)

            self.state.update_last_bar(symbol, bar["timestamp"])

            log.info("bar_processed",
                     symbol=symbol,
                     ts=bar["timestamp"].isoformat(),
                     close=bar["close"])

            self._run_strategies(symbol, ind_5m, ind_15m)

            # Higher-TF sniper: detect when 15m / 4h bars close
            bar_ts = bar["timestamp"]
            close_min = bar_ts.hour * 60 + bar_ts.minute + 5
            if close_min % 15 == 0:
                self._run_sniper_on_tf(symbol, "15m")
            if close_min % 240 == 0:
                self._run_sniper_on_tf(symbol, "4h")

        except Exception as exc:
            log.error("bar_processing_error",
                      symbol=symbol, error=str(exc), exc_info=True)

    # ----- indicator computation ----------------------------------------

    def _compute_indicators(
        self, symbol: str,
    ) -> tuple[pd.Series | None, pd.Series | None,
               pd.Series | None, pd.Series | None]:
        """
        Compute 5m (and 15m) indicators on the full bar buffer.

        Returns (raw_5m, ind_5m, raw_15m, ind_15m) where:
          raw_*  = TA column names  (for DB writes)
          ind_*  = DB column names  (for strategies)
        """
        bars = list(self._bar_buffer[symbol])
        df = pd.DataFrame(bars)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

        df_ta = compute_ta_indicators(df.copy())
        df_ta = compute_derived(df_ta)
        df_ta = detect_rsi_divergence(df_ta)
        df_ta = detect_momentum_divergence(df_ta)

        raw_5m = df_ta.iloc[-1].copy()
        funding = self._latest_funding.get(symbol, 0.0)
        raw_5m["funding_8h"] = funding
        raw_5m["funding_24h_cum"] = funding * 3
        raw_5m["liq_volume_1h"] = 0.0
        raw_5m["extras"] = pack_indicator_extras(raw_5m)

        _rn = {ta: db for ta, db in TA_COL_MAP.items() if ta in raw_5m.index}
        ind_5m = raw_5m.rename(_rn)

        raw_15m: pd.Series | None = None
        ind_15m: pd.Series | None = None
        try:
            df_15m = resample_5m_to_15m(df)
            if not df_15m.empty and len(df_15m) >= 10:
                df_15m_ta = compute_ta_indicators(df_15m.copy())
                df_15m_ta = compute_derived(df_15m_ta)
                df_15m_ta = detect_rsi_divergence(df_15m_ta)
                df_15m_ta = detect_momentum_divergence(df_15m_ta)
                raw_15m = df_15m_ta.iloc[-1].copy()
                raw_15m["extras"] = pack_indicator_extras(raw_15m)
                _rn15 = {ta: db for ta, db in TA_COL_MAP.items()
                         if ta in raw_15m.index}
                ind_15m = raw_15m.rename(_rn15)
                self._latest_15m[symbol] = ind_15m
        except Exception as exc:
            log.warning("15m_error", symbol=symbol, error=str(exc))

        if ind_15m is None:
            ind_15m = self._latest_15m.get(symbol)

        return raw_5m, ind_5m, raw_15m, ind_15m

    # ----- DB writes ----------------------------------------------------

    def _write_candle(self, bar: dict) -> None:
        if self._db_unavailable:
            return
        try:
            record = {k: v for k, v in bar.items()}
            table = Candles5m.__table__
            pk = [c.name for c in table.primary_key.columns]
            upd = [c.name for c in table.columns if c.name not in pk]
            stmt = pg_insert(table).values([record])
            stmt = stmt.on_conflict_do_update(
                index_elements=pk,
                set_={c: stmt.excluded[c] for c in upd},
            )
            with self._engine.begin() as conn:
                conn.execute(stmt)
        except Exception:
            self._db_unavailable = True
            log.warning("db_writes_disabled",
                        reason="connection failed, skipping future writes")

    def _write_indicator_row(
        self, symbol: str, row: pd.Series, model: type,
    ) -> None:
        if self._db_unavailable:
            return
        try:
            rec: dict = {
                "symbol": symbol,
                "timestamp": row.get("timestamp"),
            }
            for ta_col, db_col in TA_COL_MAP.items():
                val = row.get(ta_col)
                val = self._safe(val)
                if db_col == "supertrend_dir":
                    rec[db_col] = int(val) if val is not None else None
                else:
                    rec[db_col] = val
            for col in DERIVED_FLOAT_COLS:
                rec[col] = self._safe(row.get(col))
            rec["bb_squeeze"] = self._safe_bool(row.get("bb_squeeze"))
            ex = row.get("extras")
            rec["extras"] = ex if isinstance(ex, dict) else pack_indicator_extras(row)

            table = model.__table__
            pk = [c.name for c in table.primary_key.columns]
            upd = [c.name for c in table.columns if c.name not in pk]
            stmt = pg_insert(table).values([rec])
            stmt = stmt.on_conflict_do_update(
                index_elements=pk,
                set_={c: stmt.excluded[c] for c in upd},
            )
            with self._engine.begin() as conn:
                conn.execute(stmt)
        except Exception:
            self._db_unavailable = True

    @staticmethod
    def _safe(val: Any) -> float | None:
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        return float(val)

    @staticmethod
    def _safe_bool(val: Any) -> bool | None:
        if val is None:
            return None
        try:
            if pd.isna(val):
                return None
        except (TypeError, ValueError):
            pass
        return bool(val)

    # ----- multi-TF sniper -----------------------------------------------

    _TF_LOOKBACK = {"15m": 600, "1h": 1500, "4h": 4800}

    def _compute_tf_indicators(
        self, symbol: str, target_tf: str,
    ) -> pd.Series | None:
        """Build higher-TF indicators from the in-memory bar buffer."""
        lookback = self._TF_LOOKBACK.get(target_tf, 3000)
        try:
            buf = self._bar_buffer.get(symbol)
            if not buf or len(buf) < 100:
                return None

            bars = list(buf)[-lookback:]
            df_5m = pd.DataFrame(bars)
            if "timestamp" not in df_5m.columns or len(df_5m) < 100:
                return None

            df_5m["timestamp"] = pd.to_datetime(df_5m["timestamp"], utc=True)
            df_5m = df_5m.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
            for col in ("open", "high", "low", "close", "volume"):
                if col in df_5m.columns:
                    df_5m[col] = pd.to_numeric(df_5m[col], errors="coerce")

            resampled = resample_candles(df_5m, target_tf)
            if len(resampled) < 20:
                return None

            with_ta = compute_ta_indicators(resampled)
            with_derived = compute_derived(with_ta)
            last = with_derived.iloc[-1].copy()

            last["funding_8h"] = self._latest_funding.get(symbol, 0.0)
            last["liq_volume_1h"] = 0.0
            last["extras"] = {}

            _rn = {ta: db for ta, db in TA_COL_MAP.items()
                   if ta in last.index}
            return last.rename(_rn)
        except Exception as exc:
            log.warning("tf_indicator_error",
                        symbol=symbol, tf=target_tf, error=str(exc))
            return None

    def _run_sniper_on_tf(self, symbol: str, tf: str) -> None:
        """Run the sniper strategy on a higher TF bar that just closed."""
        sniper_map = {"15m": self._sniper_15m, "4h": self._sniper_4h}
        instances = sniper_map.get(tf)
        if instances is None:
            return

        if symbol not in instances:
            instances[symbol] = SniperStrategy()
        strategy = instances[symbol]

        trading_row = self._compute_tf_indicators(symbol, tf)
        if trading_row is None:
            log.warning("sniper_tf_no_data", symbol=symbol, tf=tf)
            return

        ctx_tf = CONTEXT_TF.get(tf, tf)
        if ctx_tf == tf:
            context_row = trading_row
        else:
            context_row = self._compute_tf_indicators(symbol, ctx_tf)
            if context_row is None:
                context_row = pd.Series(dtype=object)

        funding = self._latest_funding.get(symbol, 0.0)
        signal = strategy.generate_signal(
            symbol=symbol,
            indicators_5m=trading_row,
            indicators_15m=context_row,
            funding_rate=funding,
            liq_volume_1h=0.0,
        )

        if signal is None or signal.direction == "flat":
            log.info("sniper_eval", symbol=symbol, tf=tf, result="flat")
            return

        log.info("sniper_signal",
                 symbol=symbol, tf=tf,
                 direction=signal.direction,
                 entry=signal.entry_price,
                 sl=signal.stop_loss,
                 tp=signal.take_profit)

        self.telegram.notify_signal(
            symbol=symbol, direction=signal.direction, tf=tf,
            entry=signal.entry_price, sl=signal.stop_loss,
            tp=signal.take_profit,
        )

        positions = self.order_manager.sync_positions()
        daily_pnl = self.order_manager.get_daily_pnl()
        predicted = self._predicted_funding.get(symbol, funding)

        approved = self.risk_manager.evaluate(
            signal=signal,
            account_equity=self._equity,
            open_positions=positions,
            daily_pnl_usd=daily_pnl,
            current_funding_rate=funding,
            predicted_funding_rate=predicted,
            session=self._engine,
        )

        if approved is None:
            last = (
                self.risk_manager._blocked[-1]
                if self.risk_manager._blocked
                else {}
            )
            reason = last.get("exit_reason", "unknown")
            self.state.add_blocked({
                "symbol": symbol,
                "strategy": f"sniper_{tf}",
                "direction": signal.direction,
                "reason": reason,
                "timestamp": signal.timestamp.isoformat(),
            })
            self.telegram.notify_blocked(
                symbol=symbol, strategy=f"sniper_{tf}",
                direction=signal.direction, reason=reason,
            )
            return

        self._last_signals[symbol] = {
            "sl": approved.stop_loss,
            "tp": approved.take_profit,
            "direction": approved.direction,
            "entry": approved.entry_price,
        }

        order = self.order_manager.open_position(approved)
        log.info("sniper_executed",
                 symbol=symbol, tf=tf,
                 direction=approved.direction)
        if order:
            self.telegram.notify_order_placed(
                symbol=symbol, direction=approved.direction,
                side=order.get("side", ""),
                price=float(order.get("price", 0)),
                amount=float(order.get("amount", 0)),
                order_id=order.get("id", ""),
            )

    # ----- strategy pipeline --------------------------------------------

    def _run_strategies(
        self,
        symbol: str,
        ind_5m: pd.Series,
        ind_15m: pd.Series | None,
    ) -> None:
        funding = self._latest_funding.get(symbol, 0.0)
        liq_vol = _sf(ind_5m.get("liq_volume_1h"))
        empty_15m = pd.Series(dtype=object)

        for strategy in self._strategies:
            try:
                signal = strategy.generate_signal(
                    symbol=symbol,
                    indicators_5m=ind_5m,
                    indicators_15m=(
                        ind_15m if ind_15m is not None else empty_15m
                    ),
                    funding_rate=funding,
                    liq_volume_1h=liq_vol,
                )

                if signal is None or signal.direction == "flat":
                    continue

                positions = self.order_manager.sync_positions()
                daily_pnl = self.order_manager.get_daily_pnl()
                predicted = self._predicted_funding.get(symbol, funding)

                approved = self.risk_manager.evaluate(
                    signal=signal,
                    account_equity=self._equity,
                    open_positions=positions,
                    daily_pnl_usd=daily_pnl,
                    current_funding_rate=funding,
                    predicted_funding_rate=predicted,
                    session=self._engine,
                )

                if approved is None:
                    last = (
                        self.risk_manager._blocked[-1]
                        if self.risk_manager._blocked
                        else {}
                    )
                    self.state.add_blocked({
                        "symbol": symbol,
                        "strategy": strategy.name,
                        "direction": signal.direction,
                        "reason": last.get("exit_reason", "unknown"),
                        "timestamp": signal.timestamp.isoformat(),
                    })
                    continue

                self._last_signals[symbol] = {
                    "sl": approved.stop_loss,
                    "tp": approved.take_profit,
                    "direction": approved.direction,
                    "entry": approved.entry_price,
                }

                order = self.order_manager.open_position(approved)

                log.info("signal_executed",
                         symbol=symbol,
                         strategy=strategy.name,
                         direction=approved.direction)
                if order:
                    self.telegram.notify_order_placed(
                        symbol=symbol, direction=approved.direction,
                        side=order.get("side", ""),
                        price=float(order.get("price", 0)),
                        amount=float(order.get("amount", 0)),
                        order_id=order.get("id", ""),
                    )

            except Exception as exc:
                log.error("strategy_error",
                          symbol=symbol,
                          strategy=strategy.name,
                          error=str(exc))

    # ----- manual signal injection (called from health API) -------------

    def inject_signal(self, signal: SignalEvent) -> dict:
        """Pass a manually constructed signal through RM -> OM."""
        from src.strategies.base import SignalEvent  # noqa: F811

        positions = self.order_manager.sync_positions()
        daily_pnl = self.order_manager.get_daily_pnl()
        funding = self._latest_funding.get(signal.symbol, 0.0)

        approved = self.risk_manager.evaluate(
            signal=signal,
            account_equity=self._equity,
            open_positions=positions,
            daily_pnl_usd=daily_pnl,
            current_funding_rate=funding,
            predicted_funding_rate=funding,
            session=self._engine,
        )

        if approved is None:
            return {"status": "blocked", "reason": "risk_manager"}

        order = self.order_manager.open_position(approved)
        return {
            "status": "executed",
            "order_id": order["id"] if order else None,
        }


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main() -> None:
    """Start the live trading bot."""
    import uvicorn
    from src.api.health_api import create_health_app

    engine = create_engine(settings.sync_db_url)
    state = BotState()
    listener = WebSocketListener(engine, state)

    ws_thread = threading.Thread(target=listener.start, daemon=True)
    ws_thread.start()

    # background position-sync every 60 s
    def _sync_loop() -> None:
        while True:
            try:
                listener.order_manager.sync_positions()
            except Exception:
                pass
            time.sleep(60)

    threading.Thread(target=_sync_loop, daemon=True).start()

    app = create_health_app(engine, state, listener)
    log.info("starting_health_api", port=8080)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")


if __name__ == "__main__":
    main()
