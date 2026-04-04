"""
Order Manager — handles order placement, modification, and tracking via CCXT.

Never places an order without an approved SignalEvent.
Uses limit orders for entries (aggressive limit at best bid/ask ± 0.01%).
Only market orders are used for emergency position closes.

Connects to Bybit testnet when ``settings.bybit_testnet`` is True.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy.orm import Session

from src.config import settings
from src.db.models import TradesLog
from src.strategies.base import SignalEvent

log = structlog.get_logger()


_SYMBOL_OVERRIDES = {
    "1000PEPEUSDT": "1000PEPE/USDT:USDT",
    "1000SHIBUSDT": "1000SHIB/USDT:USDT",
    "1000FLOKIUSDT": "1000FLOKI/USDT:USDT",
    "1000BONKUSDT": "1000BONK/USDT:USDT",
    "1000LUNCUSDT": "1000LUNC/USDT:USDT",
}


def _to_ccxt_symbol(symbol: str) -> str:
    """BTCUSDT -> BTC/USDT:USDT (CCXT linear perp format)."""
    if symbol in _SYMBOL_OVERRIDES:
        return _SYMBOL_OVERRIDES[symbol]
    if symbol.endswith("USDT"):
        return f"{symbol[:-4]}/USDT:USDT"
    return symbol


class OrderManager:
    """
    Handles all order placement, modification, and tracking via CCXT.
    """

    def __init__(self, engine: Any):
        self._engine = engine
        self._exchange = None
        self._positions: list[dict] = []
        self._open_orders: dict[str, dict] = {}
        self._daily_pnl: float = 0.0
        self._init_exchange()

    def _init_exchange(self) -> None:
        if not settings.bybit_api_key or not settings.bybit_api_secret:
            log.warning("exchange_not_configured",
                        reason="BYBIT_API_KEY / BYBIT_API_SECRET empty")
            return

        import ccxt  # deferred so bot can start without ccxt for tests

        self._exchange = ccxt.bybit({
            "apiKey": settings.bybit_api_key,
            "secret": settings.bybit_api_secret,
            "options": {
                "defaultType": "linear",
                "accountType": "UNIFIED",
            },
            "enableRateLimit": True,
        })

        if settings.bybit_testnet:
            self._exchange.set_sandbox_mode(True)
        elif settings.bybit_demo:
            api_urls = self._exchange.urls.get("api", {})
            if isinstance(api_urls, dict):
                for key in api_urls:
                    if isinstance(api_urls[key], str):
                        api_urls[key] = api_urls[key].replace(
                            "api.bybit.com", "api-demo.bybit.com"
                        )
            elif isinstance(api_urls, str):
                self._exchange.urls["api"] = api_urls.replace(
                    "api.bybit.com", "api-demo.bybit.com"
                )

        self._apply_time_offset()

        try:
            self._exchange.load_markets()
            log.info("exchange_initialized",
                     testnet=settings.bybit_testnet,
                     demo=settings.bybit_demo)
        except Exception as exc:
            log.error("exchange_init_failed", error=str(exc))
            self._exchange = None

    def _apply_time_offset(self) -> None:
        """Detect local-vs-server clock drift and patch the exchange."""
        if self._exchange is None:
            return
        try:
            import json
            import time
            import urllib.request

            url = "https://api.bybit.com/v5/market/time"
            resp = urllib.request.urlopen(url, timeout=10)
            server_time = json.loads(resp.read())["time"]
            local_time = int(time.time() * 1000)
            offset = local_time - server_time

            if abs(offset) > 2000:
                original_ms = self._exchange.milliseconds
                self._exchange.milliseconds = lambda: original_ms() - offset
                log.warning("clock_offset_applied",
                            offset_ms=offset,
                            msg="System clock is off — auto-corrected")
            else:
                log.info("clock_offset_ok", offset_ms=offset)
        except Exception as exc:
            log.warning("clock_offset_check_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def open_position(self, signal: SignalEvent) -> dict | None:
        """
        Place limit entry at signal.entry_price ± 0.01 % (aggressive limit).
        Simultaneously sets SL (stop-market) and TP (limit) via params.
        Returns the raw CCXT order dict or None on failure.
        """
        if self._exchange is None:
            log.warning("open_position_skipped", reason="no exchange")
            return None

        ccxt_sym = _to_ccxt_symbol(signal.symbol)

        leverage = signal.leverage or 10
        try:
            self._exchange.set_leverage(leverage, ccxt_sym)
            log.info("leverage_set", symbol=signal.symbol, leverage=leverage)
        except Exception as exc:
            log.warning("set_leverage_failed", symbol=signal.symbol,
                        leverage=leverage, error=str(exc))

        try:
            ticker = self._exchange.fetch_ticker(ccxt_sym)
        except Exception as exc:
            log.error("ticker_fetch_failed", symbol=signal.symbol,
                      error=str(exc))
            return None

        if signal.direction == "long":
            price = ticker["ask"] * 1.0001
            side = "buy"
        else:
            price = ticker["bid"] * 0.9999
            side = "sell"

        size_usd = signal.position_size_usd or 100.0
        amount = size_usd / price if price > 0 else 0.001

        try:
            amount = float(
                self._exchange.amount_to_precision(ccxt_sym, amount)
            )
            price = float(
                self._exchange.price_to_precision(ccxt_sym, price)
            )
        except Exception:
            pass

        try:
            order = self._exchange.create_order(
                symbol=ccxt_sym,
                type="limit",
                side=side,
                amount=amount,
                price=price,
                params={
                    "stopLoss": {"triggerPrice": str(signal.stop_loss)},
                    "takeProfit": {"triggerPrice": str(signal.take_profit)},
                },
            )
        except Exception as exc:
            log.error("order_placement_failed", symbol=signal.symbol,
                      error=str(exc))
            return None

        self._open_orders[signal.symbol] = {
            "order_id": order["id"],
            "symbol": signal.symbol,
            "direction": signal.direction,
            "side": side,
            "price": price,
            "amount": amount,
            "strategy_combo": signal.strategy_combo,
            "created_at": datetime.now(timezone.utc),
        }

        self._log_entry(signal, order)

        log.info("order_placed",
                 symbol=signal.symbol,
                 order_id=order["id"],
                 side=side,
                 price=price,
                 amount=amount)
        return order

    def close_position(self, symbol: str, reason: str) -> dict | None:
        """Cancel open TP/SL orders and market-close the position."""
        if self._exchange is None:
            return None

        ccxt_sym = _to_ccxt_symbol(symbol)

        try:
            self._exchange.cancel_all_orders(ccxt_sym)
        except Exception as exc:
            log.warning("cancel_orders_failed", symbol=symbol,
                        error=str(exc))

        try:
            positions = self._exchange.fetch_positions([ccxt_sym])
        except Exception as exc:
            log.error("fetch_positions_failed", error=str(exc))
            return None

        for pos in positions:
            contracts = abs(float(pos.get("contracts", 0) or 0))
            if contracts <= 0:
                continue

            close_side = "sell" if pos["side"] == "long" else "buy"
            try:
                order = self._exchange.create_order(
                    symbol=ccxt_sym,
                    type="market",
                    side=close_side,
                    amount=contracts,
                    params={"reduceOnly": True},
                )
            except Exception as exc:
                log.error("market_close_failed", symbol=symbol,
                          error=str(exc))
                return None

            self._log_exit(symbol, order, reason)
            log.info("position_closed", symbol=symbol, reason=reason)
            self._open_orders.pop(symbol, None)
            return order

        return None

    def sync_positions(self) -> list[dict]:
        """
        Query Bybit for all open positions, reconcile with local state,
        and return current positions list.
        """
        if self._exchange is None:
            return self._positions

        try:
            raw = self._exchange.fetch_positions()
            self._positions = [
                {
                    "symbol": (p.get("info") or {}).get("symbol", ""),
                    "side": p.get("side", ""),
                    "contracts": float(p.get("contracts", 0) or 0),
                    "entryPrice": float(p.get("entryPrice", 0) or 0),
                    "unrealizedPnl": float(
                        p.get("unrealizedPnl", 0) or 0
                    ),
                    "leverage": int(p.get("leverage", 1) or 1),
                }
                for p in raw
                if abs(float(p.get("contracts", 0) or 0)) > 0
            ]
        except Exception as exc:
            log.error("position_sync_error", error=str(exc))

        return self._positions

    def handle_fill(self, fill_event: dict) -> None:
        """
        Called by WebSocket listener on execution-stream events.
        Updates trades_log with the actual fill price and computes PnL.
        """
        symbol = fill_event.get("symbol", "")
        exec_price = float(fill_event.get("execPrice", 0) or 0)
        exec_qty = float(fill_event.get("execQty", 0) or 0)
        side = fill_event.get("side", "")
        order_id = fill_event.get("orderId", "")

        log.info("fill_received",
                 symbol=symbol, price=exec_price,
                 qty=exec_qty, side=side,
                 order_id=order_id)

        tracked = self._open_orders.get(symbol)
        if not tracked:
            return

        is_closing = (
            (tracked["direction"] == "long" and side.lower() == "sell")
            or (tracked["direction"] == "short" and side.lower() == "buy")
        )

        if is_closing:
            self._update_exit_from_fill(symbol, exec_price)
            self._open_orders.pop(symbol, None)

    def get_daily_pnl(self) -> float:
        try:
            with self._engine.connect() as conn:
                from sqlalchemy import text
                today = datetime.now(timezone.utc).date().isoformat()
                row = conn.execute(
                    text(
                        "SELECT COALESCE(SUM(pnl_usd), 0) "
                        "FROM trades_log "
                        "WHERE is_backtest = FALSE "
                        "AND exit_time::date >= :today"
                    ),
                    {"today": today},
                ).scalar()
                self._daily_pnl = float(row or 0)
        except Exception:
            pass
        return self._daily_pnl

    # ------------------------------------------------------------------
    # DB logging helpers
    # ------------------------------------------------------------------

    def _log_entry(self, signal: SignalEvent, order: dict) -> None:
        try:
            with Session(self._engine) as sess:
                record = TradesLog(
                    symbol=signal.symbol,
                    direction=signal.direction,
                    leverage=signal.leverage,
                    entry_time=signal.timestamp,
                    entry_price=float(order.get("price", signal.entry_price)),
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    position_size_usd=signal.position_size_usd or 0,
                    strategy_combo=signal.strategy_combo,
                    indicators_snapshot=signal.indicators_snapshot,
                    regime_volatility=signal.regime.get("volatility"),
                    regime_funding=signal.regime.get("funding"),
                    regime_time_of_day=signal.regime.get("time_of_day"),
                    is_backtest=False,
                    notes=f"Order {order.get('id', '')}",
                )
                sess.add(record)
                sess.commit()
        except Exception as exc:
            log.error("log_entry_failed", error=str(exc))

    def _log_exit(self, symbol: str, order: dict, reason: str) -> None:
        exit_price = float(
            order.get("average", 0)
            or order.get("price", 0)
            or 0
        )
        self._update_exit_from_fill(symbol, exit_price, reason)

    def _update_exit_from_fill(
        self, symbol: str, exit_price: float, reason: str = "fill",
    ) -> None:
        try:
            with Session(self._engine) as sess:
                trade = (
                    sess.query(TradesLog)
                    .filter(
                        TradesLog.symbol == symbol,
                        TradesLog.is_backtest.is_(False),
                        TradesLog.exit_time.is_(None),
                    )
                    .order_by(TradesLog.entry_time.desc())
                    .first()
                )
                if trade is None:
                    return

                entry = float(trade.entry_price)
                if trade.direction == "long":
                    pnl_pct = (exit_price - entry) / entry if entry else 0
                else:
                    pnl_pct = (entry - exit_price) / entry if entry else 0

                trade.exit_time = datetime.now(timezone.utc)
                trade.exit_price = exit_price
                trade.exit_reason = reason
                trade.pnl_pct = pnl_pct
                trade.pnl_usd = float(trade.position_size_usd or 0) * pnl_pct
                trade.win_loss = pnl_pct > 0

                sess.commit()
                log.info("trade_exit_logged",
                         symbol=symbol, pnl_pct=f"{pnl_pct:+.4%}")
        except Exception as exc:
            log.error("log_exit_failed", error=str(exc))
