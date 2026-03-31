"""
Lightweight FastAPI app for monitoring the live trading bot.

Endpoints
---------
GET  /health           - load-balancer / systemd watchdog
GET  /status           - bot running state, uptime, WS connected
GET  /positions        - open positions with current PnL
GET  /pnl              - today / week / all-time PnL summary
GET  /top-strategies   - top 10 strategy combos by expectancy
GET  /blocked-signals  - recent blocked signals with reasons
POST /inject-signal    - push a synthetic signal through RM -> OM
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

if TYPE_CHECKING:
    from src.live.websocket_listener import BotState, WebSocketListener


class InjectPayload(BaseModel):
    symbol: str = "BTCUSDT"
    direction: str = "long"
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int = 10


def create_health_app(
    engine: Any,
    state: "BotState",
    listener: "WebSocketListener",
) -> FastAPI:
    """Build and return the FastAPI application."""

    app = FastAPI(title="Bybit Trading Bot", version="1.0.0")

    # ----- /health ------------------------------------------------------

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # ----- /status ------------------------------------------------------

    @app.get("/status")
    def status():
        uptime: float | None = None
        if state.start_time:
            uptime = (
                datetime.now(timezone.utc) - state.start_time
            ).total_seconds()
        return {
            "running": state.running,
            "uptime_seconds": uptime,
            "symbols": state.symbols,
            "ws_connected": state.ws_connected,
            "last_bar": state.last_bar,
        }

    # ----- /positions ---------------------------------------------------

    @app.get("/positions")
    def positions():
        return listener.order_manager.sync_positions()

    # ----- /pnl ---------------------------------------------------------

    @app.get("/pnl")
    def pnl():
        today = datetime.now(timezone.utc).date().isoformat()

        with engine.connect() as conn:
            row_today = conn.execute(
                text(
                    "SELECT COALESCE(SUM(pnl_usd),0), "
                    "       COUNT(*), "
                    "       SUM(CASE WHEN win_loss THEN 1 ELSE 0 END) "
                    "FROM trades_log "
                    "WHERE is_backtest = FALSE "
                    "  AND exit_time::date >= :today"
                ),
                {"today": today},
            ).first()

            row_week = conn.execute(
                text(
                    "SELECT COALESCE(SUM(pnl_usd),0), "
                    "       COUNT(*), "
                    "       SUM(CASE WHEN win_loss THEN 1 ELSE 0 END) "
                    "FROM trades_log "
                    "WHERE is_backtest = FALSE "
                    "  AND exit_time >= NOW() - INTERVAL '7 days'"
                ),
            ).first()

            row_all = conn.execute(
                text(
                    "SELECT COALESCE(SUM(pnl_usd),0), "
                    "       COUNT(*), "
                    "       SUM(CASE WHEN win_loss THEN 1 ELSE 0 END) "
                    "FROM trades_log "
                    "WHERE is_backtest = FALSE"
                ),
            ).first()

        def _summary(row):
            pnl_usd = float(row[0]) if row else 0
            trades = int(row[1]) if row else 0
            wins = int(row[2] or 0) if row else 0
            return {
                "pnl_usd": pnl_usd,
                "trades": trades,
                "win_rate": wins / trades if trades else 0,
            }

        return {
            "today": _summary(row_today),
            "week": _summary(row_week),
            "all_time": _summary(row_all),
        }

    # ----- /top-strategies ----------------------------------------------

    @app.get("/top-strategies")
    def top_strategies():
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT strategy_combo, total_trades, win_rate, "
                    "       expectancy, sharpe "
                    "FROM strategy_performance "
                    "WHERE total_trades >= 50 "
                    "ORDER BY expectancy DESC "
                    "LIMIT 10"
                ),
            ).fetchall()

        return [
            {
                "strategy_combo": list(r[0]),
                "total_trades": r[1],
                "win_rate": float(r[2]) if r[2] else 0,
                "expectancy": float(r[3]) if r[3] else 0,
                "sharpe": float(r[4]) if r[4] else 0,
            }
            for r in rows
        ]

    # ----- /blocked-signals ---------------------------------------------

    @app.get("/blocked-signals")
    def blocked_signals():
        return state.blocked_signals[-20:]

    # ----- /inject-signal -----------------------------------------------

    @app.post("/inject-signal")
    def inject_signal(payload: InjectPayload):
        """
        Push a synthetic signal through the full
        RiskManager -> OrderManager pipeline for testing.
        """
        from src.strategies.base import SignalEvent
        from src.backtest.simulator import build_indicator_snapshot

        signal = SignalEvent(
            symbol=payload.symbol,
            direction=payload.direction,
            confidence=0.5,
            entry_price=payload.entry_price,
            stop_loss=payload.stop_loss,
            take_profit=payload.take_profit,
            leverage=payload.leverage,
            indicators_snapshot={"atr_14": abs(
                payload.entry_price - payload.stop_loss
            )},
            strategy_combo=["manual_inject"],
            regime={
                "volatility": "medium",
                "funding": "neutral",
                "time_of_day": "london",
            },
            timestamp=datetime.now(timezone.utc),
        )

        try:
            result = listener.inject_signal(signal)
            return result
        except AssertionError:
            raise HTTPException(
                status_code=403,
                detail="bybit_testnet must be True",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=str(exc),
            )

    return app
