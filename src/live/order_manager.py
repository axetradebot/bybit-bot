"""
Order Manager — handles order placement, modification, and tracking via CCXT.

Never places an order without an approved SignalEvent.

Entry orders default to **post-only maker limits** with a short TTL retry
loop, falling back to a taker market only when the strategy explicitly
requested ``fill_mode="market"``.  This keeps fees on the maker side
(0.02 %) for mean-reversion / scalper strategies and matches the
backtester's hybrid fill model.

After the entry fills (any quantity), the manager:

* Recomputes SL / TP **distance** from the strategy R:R, but anchors
  them to the **actual fill price** (not the original limit price).
* Submits the partial-TP ladder from ``signal.tp_ladder`` as a series
  of reduce-only maker limit orders, sized by fraction of the filled
  quantity.
* Sets the trigger source to **LastPrice** for alts (anything except
  BTC / ETH) — Bybit's MarkPrice for alts is too noisy and triggers
  spurious stops.

Position closes are sliced when notional > ``CLOSE_SLICE_USD`` so big
exits don't blow through the book.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy.orm import Session

from src.config import settings
from src.db.models import TradesLog
from src.strategies.base import SignalEvent

log = structlog.get_logger()

# Per-side maker fee on Bybit linear (VIP0).  Used for cost reasoning;
# CCXT will report actual fees on fills.
MAKER_FEE = 0.0002
TAKER_FEE = 0.00055

# Post-only retry parameters: try N times, sleeping between attempts,
# stepping the limit slightly further from the market each time.
POST_ONLY_MAX_RETRIES = 4
POST_ONLY_RETRY_SLEEP_SEC = 2.0
POST_ONLY_STEP_BPS = 0.0002        # 2 bps step away from mid each retry
POST_ONLY_FALLBACK_TO_TAKER = True

# Sliced market-close: chunk into ~$5k notional pieces for liquidity.
CLOSE_SLICE_USD = 5_000.0
CLOSE_SLICE_SLEEP_SEC = 0.4

# Symbols that use MarkPrice safely.  Everything else gets LastPrice.
_MARK_PRICE_SAFE = frozenset({"BTCUSDT", "ETHUSDT"})


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

    @staticmethod
    def _trigger_source(symbol: str) -> str:
        """
        ``MarkPrice`` for BTC/ETH (deep books, well-behaved mark), else
        ``LastPrice`` for alts to avoid mark-price noise stops.
        """
        return (
            "MarkPrice" if symbol in _MARK_PRICE_SAFE else "LastPrice"
        )

    def _round_price(self, ccxt_sym: str, price: float) -> float:
        try:
            return float(
                self._exchange.price_to_precision(ccxt_sym, price)
            )
        except Exception:
            return price

    def _round_amount(self, ccxt_sym: str, amount: float) -> float:
        try:
            return float(
                self._exchange.amount_to_precision(ccxt_sym, amount)
            )
        except Exception:
            return amount

    def open_position(self, signal: SignalEvent) -> dict | None:
        """
        Place a post-only maker entry at ``signal.entry_price`` (with
        TTL retries) and, on fill, attach SL / TP / partial-TP-ladder
        anchored to the actual fill price.
        """
        if self._exchange is None:
            log.warning("open_position_skipped", reason="no exchange")
            return None

        ccxt_sym = _to_ccxt_symbol(signal.symbol)

        leverage = signal.leverage or 20
        try:
            self._exchange.set_margin_mode("isolated", ccxt_sym)
        except Exception as exc:
            log.warning("set_margin_mode_failed", symbol=signal.symbol,
                        error=str(exc))
        try:
            self._exchange.set_leverage(leverage, ccxt_sym)
            log.info("leverage_set", symbol=signal.symbol,
                     leverage=leverage, margin="isolated")
        except Exception as exc:
            log.warning("set_leverage_failed", symbol=signal.symbol,
                        leverage=leverage, error=str(exc))

        # ---- pricing ---------------------------------------------------
        try:
            ticker = self._exchange.fetch_ticker(ccxt_sym)
        except Exception as exc:
            log.error("ticker_fetch_failed", symbol=signal.symbol,
                      error=str(exc))
            return None

        bid = float(ticker.get("bid") or 0)
        ask = float(ticker.get("ask") or 0)
        side = "buy" if signal.direction == "long" else "sell"

        # Post-only ⇒ price must NOT cross the spread.  Anchor at signal
        # price but clip to the safe (passive) side of the book.
        anchor = float(signal.entry_price)
        if side == "buy":
            limit = min(anchor, bid) if bid > 0 else anchor
        else:
            limit = max(anchor, ask) if ask > 0 else anchor

        size_usd = signal.position_size_usd or 100.0
        amount = size_usd / limit if limit > 0 else 0.001
        amount = self._round_amount(ccxt_sym, amount)

        fill_mode = signal.effective_fill_mode() or "post_only"
        order = self._submit_entry_with_retries(
            ccxt_sym=ccxt_sym,
            signal=signal,
            side=side,
            amount=amount,
            limit_price=limit,
            fill_mode=fill_mode,
        )
        if order is None:
            return None

        # ---- rebase SL / TP off the actual fill -----------------------
        avg_fill = float(
            order.get("average") or order.get("price") or limit
        )
        filled_qty = float(
            order.get("filled") or order.get("amount") or amount
        )
        if filled_qty <= 0:
            log.warning("entry_unfilled_after_retries",
                        symbol=signal.symbol, order_id=order.get("id"))
            return order

        sl_price, tp_price = self._rebase_sl_tp(signal, avg_fill)
        self._attach_protective_orders(
            ccxt_sym=ccxt_sym,
            symbol=signal.symbol,
            side=side,
            signal=signal,
            avg_fill=avg_fill,
            filled_qty=filled_qty,
            sl_price=sl_price,
            tp_price=tp_price,
        )

        self._open_orders[signal.symbol] = {
            "order_id": order["id"],
            "symbol": signal.symbol,
            "direction": signal.direction,
            "side": side,
            "price": avg_fill,
            "amount": filled_qty,
            "stop_loss": sl_price,
            "take_profit": tp_price,
            "strategy_combo": signal.strategy_combo,
            "created_at": datetime.now(timezone.utc),
        }

        self._log_entry(signal, order)

        log.info("order_placed",
                 symbol=signal.symbol,
                 order_id=order["id"],
                 side=side,
                 fill_mode=fill_mode,
                 fill_price=avg_fill,
                 amount=filled_qty,
                 stop_loss=sl_price,
                 take_profit=tp_price)
        return order

    # ------------------------------------------------------------------
    # Entry retry loop
    # ------------------------------------------------------------------

    def _submit_entry_with_retries(
        self,
        *,
        ccxt_sym: str,
        signal: SignalEvent,
        side: str,
        amount: float,
        limit_price: float,
        fill_mode: str,
    ) -> dict | None:
        """
        Attempt up to ``POST_ONLY_MAX_RETRIES`` post-only placements.
        Cancel-and-replace one tick further from market each retry.
        Optionally fall back to taker market on TTL expiry.
        """
        if fill_mode == "market":
            return self._place_market_entry(
                ccxt_sym=ccxt_sym, signal=signal, side=side, amount=amount,
            )

        last_error: str | None = None
        last_order: dict | None = None
        price = self._round_price(ccxt_sym, limit_price)

        for attempt in range(POST_ONLY_MAX_RETRIES):
            try:
                order = self._exchange.create_order(
                    symbol=ccxt_sym,
                    type="limit",
                    side=side,
                    amount=amount,
                    price=price,
                    params={
                        "timeInForce": "PostOnly",
                        "reduceOnly": False,
                    },
                )
            except Exception as exc:
                last_error = str(exc)
                log.warning("post_only_place_failed",
                            symbol=signal.symbol,
                            attempt=attempt + 1,
                            price=price,
                            error=last_error)
                # If exchange rejected because price would cross,
                # back off the price one step.
                price = self._step_passive(side, price)
                price = self._round_price(ccxt_sym, price)
                time.sleep(POST_ONLY_RETRY_SLEEP_SEC)
                continue

            last_order = order
            order_id = order.get("id", "")
            time.sleep(POST_ONLY_RETRY_SLEEP_SEC)

            # Poll the order status to see if it filled.
            try:
                fetched = self._exchange.fetch_order(order_id, ccxt_sym)
            except Exception:
                fetched = order
            status = (fetched.get("status") or "").lower()
            filled = float(fetched.get("filled") or 0)
            if filled > 0:
                return fetched
            if status in ("closed", "filled"):
                return fetched

            # Still open and unfilled — cancel and step further out.
            try:
                self._exchange.cancel_order(order_id, ccxt_sym)
            except Exception:
                pass
            price = self._step_passive(side, price)
            price = self._round_price(ccxt_sym, price)

        if POST_ONLY_FALLBACK_TO_TAKER:
            log.warning("post_only_ttl_expired_falling_back_taker",
                        symbol=signal.symbol,
                        attempts=POST_ONLY_MAX_RETRIES,
                        last_error=last_error)
            return self._place_market_entry(
                ccxt_sym=ccxt_sym, signal=signal, side=side, amount=amount,
            )
        log.error("entry_failed",
                  symbol=signal.symbol, last_error=last_error)
        return last_order

    def _place_market_entry(
        self, *, ccxt_sym: str, signal: SignalEvent, side: str,
        amount: float,
    ) -> dict | None:
        try:
            return self._exchange.create_order(
                symbol=ccxt_sym,
                type="market",
                side=side,
                amount=amount,
            )
        except Exception as exc:
            log.error("market_entry_failed",
                      symbol=signal.symbol, error=str(exc))
            return None

    @staticmethod
    def _step_passive(side: str, price: float) -> float:
        """Step the limit further into the passive side of the book."""
        if side == "buy":
            return price * (1.0 - POST_ONLY_STEP_BPS)
        return price * (1.0 + POST_ONLY_STEP_BPS)

    # ------------------------------------------------------------------
    # SL / TP rebase + ladder placement
    # ------------------------------------------------------------------

    @staticmethod
    def _rebase_sl_tp(
        signal: SignalEvent, avg_fill: float,
    ) -> tuple[float, float]:
        """
        Translate the strategy's SL/TP distances onto the *actual* fill
        price.  Preserves R:R; protects against fill-price drift on
        post-only orders that get worked away from the signal price.
        """
        sl_dist = abs(float(signal.entry_price) - float(signal.stop_loss))
        tp_dist = abs(float(signal.take_profit) - float(signal.entry_price))
        if signal.direction == "long":
            return avg_fill - sl_dist, avg_fill + tp_dist
        return avg_fill + sl_dist, avg_fill - tp_dist

    def _attach_protective_orders(
        self,
        *,
        ccxt_sym: str,
        symbol: str,
        side: str,
        signal: SignalEvent,
        avg_fill: float,
        filled_qty: float,
        sl_price: float,
        tp_price: float,
    ) -> None:
        """
        Set position-level SL via Bybit's trading-stop endpoint and post
        a partial-TP ladder of reduce-only maker limits.  The remaining
        un-laddered quantity rides to the absolute ``tp_price``.
        """
        trigger = self._trigger_source(symbol)
        sl_str = str(self._round_price(ccxt_sym, sl_price))
        try:
            self._exchange.private_post_v5_position_trading_stop({
                "category": "linear",
                "symbol": symbol,
                "stopLoss": sl_str,
                "slTriggerBy": trigger,
                "positionIdx": 0,
            })
            log.info("stop_loss_attached",
                     symbol=symbol, sl=sl_str, trigger=trigger)
        except Exception as exc:
            log.error("attach_sl_failed", symbol=symbol, error=str(exc))

        ladder = signal.tp_ladder or []
        close_side = "sell" if side == "buy" else "buy"
        sl_dist = abs(avg_fill - sl_price)
        sign = 1 if signal.direction == "long" else -1

        ladder_qty_used = 0.0
        for r_mult, frac in ladder:
            rung_price = avg_fill + sign * r_mult * sl_dist
            rung_qty = self._round_amount(ccxt_sym, filled_qty * frac)
            if rung_qty <= 0:
                continue
            try:
                self._exchange.create_order(
                    symbol=ccxt_sym,
                    type="limit",
                    side=close_side,
                    amount=rung_qty,
                    price=self._round_price(ccxt_sym, rung_price),
                    params={
                        "timeInForce": "PostOnly",
                        "reduceOnly": True,
                    },
                )
                ladder_qty_used += rung_qty
                log.info("tp_ladder_rung_placed", symbol=symbol,
                         r=r_mult, qty=rung_qty, price=rung_price)
            except Exception as exc:
                log.warning("tp_ladder_rung_failed",
                            symbol=symbol, r=r_mult, error=str(exc))

        # Remaining quantity rides to absolute TP.
        remaining = max(0.0, filled_qty - ladder_qty_used)
        remaining = self._round_amount(ccxt_sym, remaining)
        if remaining > 0:
            try:
                self._exchange.create_order(
                    symbol=ccxt_sym,
                    type="limit",
                    side=close_side,
                    amount=remaining,
                    price=self._round_price(ccxt_sym, tp_price),
                    params={
                        "timeInForce": "PostOnly",
                        "reduceOnly": True,
                    },
                )
                log.info("tp_terminal_placed",
                         symbol=symbol, qty=remaining, price=tp_price)
            except Exception as exc:
                log.warning("tp_terminal_failed",
                            symbol=symbol, error=str(exc))

    def update_stop_loss(self, symbol: str, new_sl: float) -> bool:
        """Move the stop-loss for an open position on Bybit."""
        if self._exchange is None:
            return False
        ccxt_sym = _to_ccxt_symbol(symbol)
        trigger = self._trigger_source(symbol)
        try:
            new_sl_str = str(
                float(self._exchange.price_to_precision(ccxt_sym, new_sl))
            )
        except Exception:
            new_sl_str = str(new_sl)
        try:
            self._exchange.set_trading_stop(ccxt_sym, params={
                "stopLoss": new_sl_str,
                "slTriggerBy": trigger,
                "positionIdx": 0,
            })
            log.info("stop_loss_updated", symbol=symbol,
                     new_sl=new_sl_str, trigger=trigger)
            return True
        except AttributeError:
            pass
        try:
            self._exchange.private_post_v5_position_trading_stop({
                "category": "linear",
                "symbol": symbol,
                "stopLoss": new_sl_str,
                "slTriggerBy": trigger,
                "positionIdx": 0,
            })
            log.info("stop_loss_updated", symbol=symbol,
                     new_sl=new_sl_str, trigger=trigger)
            return True
        except Exception as exc:
            log.error("update_sl_failed", symbol=symbol, error=str(exc))
            return False

    def rebase_protective_orders_from_fill(
        self, signal: SignalEvent, fill_price: float, filled_qty: float,
    ) -> None:
        """
        Re-anchor SL/TP/ladder onto the actual fill price.  Called by
        the WebSocket fill handler when the entry executes (possibly
        with non-trivial slippage vs the original limit).
        """
        if self._exchange is None:
            return
        ccxt_sym = _to_ccxt_symbol(signal.symbol)
        side = "buy" if signal.direction == "long" else "sell"
        sl_price, tp_price = self._rebase_sl_tp(signal, fill_price)
        # Cancel stale resting reduce-only orders before re-attaching.
        try:
            self._exchange.cancel_all_orders(
                ccxt_sym, params={"orderFilter": "Order"},
            )
        except Exception as exc:
            log.warning("cancel_resting_failed",
                        symbol=signal.symbol, error=str(exc))
        self._attach_protective_orders(
            ccxt_sym=ccxt_sym,
            symbol=signal.symbol,
            side=side,
            signal=signal,
            avg_fill=fill_price,
            filled_qty=filled_qty,
            sl_price=sl_price,
            tp_price=tp_price,
        )

    def close_position(self, symbol: str, reason: str) -> dict | None:
        """
        Cancel open TP/SL orders and close the position.  Notional larger
        than ``CLOSE_SLICE_USD`` is split into market chunks paced by
        ``CLOSE_SLICE_SLEEP_SEC`` to reduce single-tick slippage.
        """
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

        last_order: dict | None = None
        for pos in positions:
            contracts = abs(float(pos.get("contracts", 0) or 0))
            if contracts <= 0:
                continue
            mark = float(pos.get("markPrice") or pos.get("entryPrice") or 0)
            close_side = "sell" if pos["side"] == "long" else "buy"

            slices = self._size_close_slices(contracts, mark, ccxt_sym)
            for idx, slice_qty in enumerate(slices):
                try:
                    order = self._exchange.create_order(
                        symbol=ccxt_sym,
                        type="market",
                        side=close_side,
                        amount=slice_qty,
                        params={"reduceOnly": True},
                    )
                    last_order = order
                except Exception as exc:
                    log.error("market_close_failed",
                              symbol=symbol, slice=idx,
                              qty=slice_qty, error=str(exc))
                    break
                if idx + 1 < len(slices):
                    time.sleep(CLOSE_SLICE_SLEEP_SEC)

        if last_order is not None:
            self._log_exit(symbol, last_order, reason)
            log.info("position_closed", symbol=symbol, reason=reason,
                     slices=len(slices) if 'slices' in locals() else 1)
        self._open_orders.pop(symbol, None)
        return last_order

    def _size_close_slices(
        self, contracts: float, mark_price: float, ccxt_sym: str,
    ) -> list[float]:
        """Break a close into chunks of <= ``CLOSE_SLICE_USD`` notional."""
        if mark_price <= 0:
            return [self._round_amount(ccxt_sym, contracts)]
        notional = contracts * mark_price
        if notional <= CLOSE_SLICE_USD:
            return [self._round_amount(ccxt_sym, contracts)]
        n_slices = int(notional // CLOSE_SLICE_USD) + 1
        per_slice = contracts / n_slices
        slices = [self._round_amount(ccxt_sym, per_slice)] * (n_slices - 1)
        already = sum(slices)
        tail = self._round_amount(ccxt_sym, max(0.0, contracts - already))
        if tail > 0:
            slices.append(tail)
        return [s for s in slices if s > 0]

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
                    entry_price=float(order.get("price") or signal.entry_price),
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
