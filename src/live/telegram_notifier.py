"""
Telegram notification service for the trading bot.

Sends trade fills, signal alerts, blocked signals, and hourly status
updates to a Telegram chat via the Bot API.

Setup
-----
1. Message @BotFather on Telegram -> /newbot -> copy the token
2. Send any message to your bot, then visit:
   https://api.telegram.org/bot<TOKEN>/getUpdates
   to find your chat_id
3. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any

import structlog
import urllib.request
import urllib.parse
import json

from src.config import settings

log = structlog.get_logger()


class TelegramNotifier:
    """Async-safe Telegram bot that sends messages via HTTP."""

    def __init__(self) -> None:
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._enabled = bool(self._token and self._chat_id)
        self._base_url = f"https://api.telegram.org/bot{self._token}"

        self._trades_session: list[dict] = []
        self._signals_session: list[dict] = []
        self._blocked_session: list[dict] = []
        self._pending_fills: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._start_time = datetime.now(timezone.utc)

        if self._enabled:
            log.info("telegram_enabled", chat_id=self._chat_id[:4] + "...")
        else:
            log.info("telegram_disabled", reason="token or chat_id not set")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        if not self._enabled:
            return False
        try:
            payload = {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self._base_url}/sendMessage",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                if not result.get("ok"):
                    log.warning("telegram_send_failed", result=result)
                    return False
            return True
        except Exception as exc:
            log.warning("telegram_error", error=str(exc))
            return False

    def _send_async(self, text: str) -> None:
        """Fire-and-forget send in a background thread."""
        threading.Thread(
            target=self.send_message, args=(text,), daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Event hooks — called from the listener / order manager
    # ------------------------------------------------------------------

    def notify_startup(self, symbols: list[str], equity: float,
                       risk_pct: float, strategy: str) -> None:
        msg = (
            "<b>Bot Started</b>\n"
            f"Strategy: <code>{strategy}</code>\n"
            f"Symbols: <code>{', '.join(symbols)}</code>\n"
            f"Equity: <code>${equity:,.2f}</code>\n"
            f"Risk/trade: <code>{risk_pct:.1%}</code>\n"
            f"Timeframes: <code>15m, 4h</code>\n"
            f"Time: <code>{datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}</code>"
        )
        self._send_async(msg)

    def notify_signal(self, symbol: str, direction: str, tf: str,
                      entry: float, sl: float, tp: float,
                      strategy: str = "sniper") -> None:
        arrow = "\u2B06" if direction == "long" else "\u2B07"
        rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
        msg = (
            f"{arrow} <b>Signal: {symbol} {direction.upper()}</b>\n"
            f"TF: <code>{tf}</code> | Strategy: <code>{strategy}</code>\n"
            f"Entry: <code>{entry:.6g}</code>\n"
            f"SL: <code>{sl:.6g}</code> | TP: <code>{tp:.6g}</code>\n"
            f"R:R: <code>1:{rr:.1f}</code>\n"
            f"Time: <code>{datetime.now(timezone.utc):%H:%M UTC}</code>"
        )
        with self._lock:
            self._signals_session.append({
                "symbol": symbol, "direction": direction,
                "tf": tf, "time": datetime.now(timezone.utc),
            })
        self._send_async(msg)

    def notify_order_placed(self, symbol: str, direction: str,
                            side: str, price: float, amount: float,
                            order_id: str) -> None:
        arrow = "\u2B06" if direction == "long" else "\u2B07"
        msg = (
            f"{arrow} <b>Order Placed: {symbol}</b>\n"
            f"Side: <code>{side}</code> | Price: <code>{price:.6g}</code>\n"
            f"Size: <code>{amount:.6g}</code>\n"
            f"Order ID: <code>{order_id[:12]}...</code>\n"
            f"Time: <code>{datetime.now(timezone.utc):%H:%M UTC}</code>"
        )
        with self._lock:
            self._trades_session.append({
                "symbol": symbol, "direction": direction,
                "price": price, "amount": amount,
                "time": datetime.now(timezone.utc),
            })
        self._send_async(msg)

    def notify_fill(self, symbol: str, side: str, price: float,
                    qty: float, order_id: str,
                    sl: float = 0, tp: float = 0,
                    is_close: bool = False,
                    entry_price: float = 0,
                    direction: str = "") -> None:
        """Aggregate partial fills per order_id, send one message after 3s."""
        with self._lock:
            if order_id not in self._pending_fills:
                self._pending_fills[order_id] = {
                    "symbol": symbol, "side": side, "direction": direction,
                    "total_qty": 0.0, "total_cost": 0.0,
                    "sl": sl, "tp": tp, "is_close": is_close,
                    "entry_price": entry_price,
                }
            pf = self._pending_fills[order_id]
            pf["total_qty"] += qty
            pf["total_cost"] += price * qty

        def _delayed_send():
            time.sleep(3)
            with self._lock:
                data = self._pending_fills.pop(order_id, None)
            if not data or data["total_qty"] == 0:
                return
            avg_price = data["total_cost"] / data["total_qty"]
            self._send_fill_summary(data, avg_price, order_id)

        threading.Thread(target=_delayed_send, daemon=True).start()

    def _send_fill_summary(self, data: dict, avg_price: float,
                           order_id: str) -> None:
        sym = data["symbol"]
        side = data["side"]
        qty = data["total_qty"]
        direction = data["direction"]

        if data["is_close"]:
            entry = data["entry_price"]
            if entry and entry > 0:
                if direction == "long":
                    pnl_pct = (avg_price - entry) / entry
                else:
                    pnl_pct = (entry - avg_price) / entry
                pnl_usd = qty * avg_price * pnl_pct
                emoji = "\u2705" if pnl_pct > 0 else "\u274C"
                result = "TP Hit" if pnl_pct > 0 else "SL Hit"
            else:
                pnl_pct = 0
                pnl_usd = 0
                emoji = "\u2139"
                result = "Closed"

            msg = (
                f"{emoji} <b>{result}: {sym}</b>\n"
                f"Direction: <code>{direction.upper()}</code>\n"
                f"Exit Price: <code>{avg_price:.6g}</code>\n"
                f"Entry was: <code>{entry:.6g}</code>\n"
                f"Size: <code>{qty:.6g}</code>\n"
                f"PnL: <code>{pnl_pct:+.2%}</code>"
            )
            if pnl_usd:
                msg += f" (<code>${pnl_usd:+,.2f}</code>)"
        else:
            arrow = "\u2B06" if side.lower() == "buy" else "\u2B07"
            msg = (
                f"{arrow} <b>Position Opened: {sym}</b>\n"
                f"Direction: <code>{direction.upper() or side}</code>\n"
                f"Entry: <code>{avg_price:.6g}</code>\n"
                f"Size: <code>{qty:.6g}</code>\n"
                f"SL: <code>{data['sl']:.6g}</code> | "
                f"TP: <code>{data['tp']:.6g}</code>"
            )

        self._send_async(msg)

    def notify_trade_closed(self, symbol: str, direction: str,
                            pnl_pct: float, pnl_usd: float = 0) -> None:
        emoji = "\u2705" if pnl_pct > 0 else "\u274C"
        msg = (
            f"{emoji} <b>Trade Closed: {symbol}</b>\n"
            f"Direction: <code>{direction}</code>\n"
            f"PnL: <code>{pnl_pct:+.2%}</code>"
        )
        if pnl_usd:
            msg += f" (<code>${pnl_usd:+,.2f}</code>)"
        self._send_async(msg)

    def notify_blocked(self, symbol: str, strategy: str,
                       direction: str, reason: str) -> None:
        with self._lock:
            self._blocked_session.append({
                "symbol": symbol, "reason": reason,
                "time": datetime.now(timezone.utc),
            })

    def notify_error(self, context: str, error: str) -> None:
        msg = (
            f"\u26A0 <b>Error: {context}</b>\n"
            f"<code>{error[:500]}</code>"
        )
        self._send_async(msg)

    # ------------------------------------------------------------------
    # Hourly status report
    # ------------------------------------------------------------------

    def send_hourly_status(self, exchange: Any = None) -> None:
        now = datetime.now(timezone.utc)
        uptime = now - self._start_time
        hours = int(uptime.total_seconds() // 3600)
        mins = int((uptime.total_seconds() % 3600) // 60)

        balance_str = "N/A"
        positions_str = "None"
        pnl_str = "N/A"

        if exchange is not None:
            try:
                bal = exchange.fetch_balance()
                usdt = bal.get("USDT", {})
                total = float(usdt.get("total", 0))
                free = float(usdt.get("free", 0))
                balance_str = f"${total:,.2f} (free: ${free:,.2f})"
            except Exception:
                pass

            try:
                raw = exchange.fetch_positions()
                open_pos = [
                    p for p in raw
                    if abs(float(p.get("contracts", 0) or 0)) > 0
                ]
                if open_pos:
                    lines = []
                    total_upnl = 0.0
                    for p in open_pos:
                        sym = (p.get("info") or {}).get("symbol", "?")
                        side = p.get("side", "?")
                        upnl = float(p.get("unrealizedPnl", 0) or 0)
                        total_upnl += upnl
                        entry = float(p.get("entryPrice", 0) or 0)
                        lines.append(
                            f"  {sym} {side} @ {entry:.6g} "
                            f"(uPnL: ${upnl:+,.2f})"
                        )
                    positions_str = "\n".join(lines)
                    pnl_str = f"${total_upnl:+,.2f}"
                else:
                    positions_str = "None"
                    pnl_str = "$0.00"
            except Exception:
                pass

        with self._lock:
            n_signals = len(self._signals_session)
            n_trades = len(self._trades_session)
            n_blocked = len(self._blocked_session)

        msg = (
            f"<b>Hourly Status</b>\n"
            f"Time: <code>{now:%Y-%m-%d %H:%M UTC}</code>\n"
            f"Uptime: <code>{hours}h {mins}m</code>\n\n"
            f"<b>Balance:</b> <code>{balance_str}</code>\n"
            f"<b>Unrealized PnL:</b> <code>{pnl_str}</code>\n\n"
            f"<b>Open Positions:</b>\n"
            f"<code>{positions_str}</code>\n\n"
            f"<b>Session Stats:</b>\n"
            f"  Signals: <code>{n_signals}</code>\n"
            f"  Orders placed: <code>{n_trades}</code>\n"
            f"  Blocked: <code>{n_blocked}</code>"
        )
        self._send_async(msg)

    # ------------------------------------------------------------------
    # Background hourly loop
    # ------------------------------------------------------------------

    def start_hourly_loop(self, exchange: Any = None) -> None:
        """Start a daemon thread that sends status every hour.

        No-op unless both the bot is enabled AND
        ``settings.telegram_hourly_status`` is true.  Trade fills,
        signal alerts, and blocked-signal notifications are unaffected
        and continue firing as long as Telegram credentials are set.
        """
        if not self._enabled:
            return
        if not settings.telegram_hourly_status:
            log.info("telegram_hourly_loop_disabled",
                     reason="TELEGRAM_HOURLY_STATUS=false")
            return

        def _loop():
            while True:
                next_hour = 3600 - (time.time() % 3600)
                time.sleep(next_hour)
                try:
                    self.send_hourly_status(exchange)
                except Exception as exc:
                    log.warning("hourly_status_error", error=str(exc))

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        log.info("telegram_hourly_loop_started")
