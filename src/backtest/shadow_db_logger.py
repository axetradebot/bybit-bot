"""
Shadow DB logger — writes every closed trade to trades_log.

The indicators_snapshot JSONB is self-contained: if indicators_5m were
deleted, trades_log alone is sufficient to rebuild strategy_performance.
"""

from __future__ import annotations

from datetime import datetime

import structlog
from sqlalchemy import Engine, delete, and_
from sqlalchemy.orm import Session

from src.db.models import TradesLog
from src.backtest.simulator import ClosedTrade

log = structlog.get_logger()


class ShadowDBLogger:

    def __init__(self, engine: Engine):
        self.engine = engine

    def clear_backtest_trades(
        self,
        strategy_combo: list[str],
        symbol: str,
        from_date: str,
        to_date: str,
    ) -> int:
        """Delete previous backtest trades for this strategy+symbol+range."""
        with Session(self.engine) as session:
            stmt = delete(TradesLog).where(
                and_(
                    TradesLog.is_backtest.is_(True),
                    TradesLog.symbol == symbol,
                    TradesLog.strategy_combo == strategy_combo,
                    TradesLog.entry_time >= from_date,
                    TradesLog.entry_time < to_date,
                )
            )
            result = session.execute(stmt)
            session.commit()
            deleted = result.rowcount
            if deleted > 0:
                log.info(
                    "cleared_old_backtest_trades",
                    symbol=symbol,
                    strategy=strategy_combo,
                    deleted=deleted,
                )
            return deleted

    def log_trades(self, trades: list[ClosedTrade]) -> int:
        """Insert all closed trades into trades_log. Returns count inserted."""
        if not trades:
            return 0

        with Session(self.engine) as session:
            for trade in trades:
                record = TradesLog(
                    symbol=trade.symbol,
                    direction=trade.direction,
                    leverage=trade.leverage,
                    entry_time=trade.entry_time,
                    exit_time=trade.exit_time,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    stop_loss=trade.stop_loss,
                    take_profit=trade.take_profit,
                    position_size_usd=trade.position_size_usd,
                    pnl_pct=trade.pnl_pct,
                    pnl_usd=trade.pnl_usd,
                    funding_paid_usd=trade.funding_paid_usd,
                    win_loss=trade.win_loss,
                    strategy_combo=trade.strategy_combo,
                    indicators_snapshot=trade.indicators_snapshot,
                    regime_volatility=trade.regime_volatility,
                    regime_funding=trade.regime_funding,
                    regime_time_of_day=trade.regime_time_of_day,
                    exit_reason=trade.exit_reason,
                    is_backtest=True,
                    notes=trade.notes,
                )
                session.add(record)
            session.commit()

        log.info("trades_logged", count=len(trades))
        return len(trades)
