"""
Risk Manager — hard gate between strategy signals and order execution.

Every signal must pass through ``evaluate()`` before being acted on.
Blocked signals are collected in memory and flushed to ``trades_log``
via ``log_blocked_signals()`` so they remain visible in analytics.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import select, text as sa_text
from sqlalchemy.orm import Session

from src.db.models import StrategyPerformance, TradesLog
from src.strategies.base import SignalEvent, _sf

log = structlog.get_logger()


class RiskManager:
    """
    Hard gate between strategy signals and order execution.
    Every signal must pass through evaluate() before being acted on.

    A blocked signal is collected (not immediately persisted) and can
    be flushed to trades_log via log_blocked_signals() with
    win_loss=NULL, exit_reason set to the gate name, and pnl_pct=0.
    This ensures blocked signals are visible in analytics — you want
    to know if the funding gate is firing constantly.
    """

    def __init__(self, is_backtest: bool = True, risk_pct: float = 0.01):
        self.is_backtest = is_backtest
        self.risk_pct = risk_pct
        self._daily_loss_halted = False
        self._halt_date: date | None = None
        self._blocked: list[dict] = []

    def evaluate(
        self,
        signal: SignalEvent,
        account_equity: float,
        open_positions: list[dict],
        daily_pnl_usd: float,
        current_funding_rate: float,
        predicted_funding_rate: float,
        session: Any = None,
    ) -> SignalEvent | None:
        """
        Run all gates in order.  Return approved (possibly modified)
        SignalEvent or None if blocked.

        Gates run in this exact order — first block wins:
          1. daily_loss_gate
          2. session_open_gate
          3. max_positions_gate
          4. funding_gate
          5. leverage_gate        (modifies leverage, never blocks)
          6. position_size_gate   (sets position_size_usd on signal)
          7. rr_gate              (blocks if R:R < 1.5)
          8. liquidation_cluster_gate (reduces size if extreme)
        """
        approved = signal.model_copy(deep=True)

        sig_date = (
            signal.timestamp.date()
            if hasattr(signal.timestamp, "date")
            else date.today()
        )

        if self._halt_date is not None and self._halt_date < sig_date:
            self._daily_loss_halted = False
            self._halt_date = None

        # Gate 1
        if self._daily_loss_halted:
            self._record_blocked(signal, "daily_loss_gate")
            return None
        if self._daily_loss_gate(account_equity, daily_pnl_usd):
            self._daily_loss_halted = True
            self._halt_date = sig_date
            self._record_blocked(signal, "daily_loss_gate")
            return None

        # Gate 2
        if self._session_open_gate(signal.timestamp):
            self._record_blocked(signal, "session_gate")
            return None

        # Gate 3
        if self._max_positions_gate(signal.symbol, open_positions):
            self._record_blocked(signal, "max_positions_gate")
            return None

        # Gate 4
        if self._funding_gate(
            signal.direction, current_funding_rate, predicted_funding_rate,
        ):
            self._record_blocked(signal, "funding_gate")
            return None

        # Gate 5 — modifies leverage, never blocks
        self._leverage_gate(approved, session)

        # Gate 6 — may block
        if self._position_size_gate(approved, account_equity):
            self._record_blocked(signal, "liquidation_too_close")
            return None

        # Gate 7
        if self._rr_gate(approved):
            self._record_blocked(signal, "rr_gate")
            return None

        # Gate 8 — modifies size, never blocks
        self._liquidation_cluster_gate(approved, session)

        return approved

    # ------------------------------------------------------------------
    # Gate implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _daily_loss_gate(account_equity: float, daily_pnl_usd: float) -> bool:
        return daily_pnl_usd < -(account_equity * 0.05)

    @staticmethod
    def _session_open_gate(ts: datetime) -> bool:
        """Block if within 5 minutes of 00:00, 08:00, or 16:00 UTC."""
        minute_of_day = ts.hour * 60 + ts.minute
        for boundary in (0, 480, 960):
            diff = abs(minute_of_day - boundary)
            diff = min(diff, 1440 - diff)
            if diff <= 5:
                return True
        return False

    @staticmethod
    def _max_positions_gate(
        symbol: str, open_positions: list[dict],
    ) -> bool:
        if len(open_positions) >= 3:
            return True
        for pos in open_positions:
            if pos.get("symbol") == symbol:
                return True
        return False

    @staticmethod
    def _funding_gate(
        direction: str,
        current_rate: float,
        predicted_rate: float,
    ) -> bool:
        if direction == "long":
            return current_rate > 0.0005 or predicted_rate > 0.0008
        if direction == "short":
            return current_rate < -0.0005 or predicted_rate < -0.0008
        return False

    @staticmethod
    def _leverage_gate(signal: SignalEvent, engine: Any = None) -> None:
        total_trades = 0
        win_rate = 0.0

        if engine is not None:
            try:
                with Session(engine) as sess:
                    stmt = (
                        select(
                            StrategyPerformance.win_rate,
                            StrategyPerformance.total_trades,
                        )
                        .where(
                            StrategyPerformance.strategy_combo
                            == signal.strategy_combo,
                        )
                        .limit(1)
                    )
                    row = sess.execute(stmt).first()
                    if row:
                        win_rate = float(row[0] or 0)
                        total_trades = int(row[1] or 0)
            except Exception:
                pass

        if total_trades >= 300 and win_rate >= 0.62:
            max_lev = 20
        elif total_trades >= 100 and win_rate >= 0.58:
            max_lev = 15
        else:
            max_lev = 10

        max_lev = min(max_lev, 25)
        signal.leverage = min(signal.leverage, max_lev)

    def _position_size_gate(
        self, signal: SignalEvent, account_equity: float,
    ) -> bool:
        entry = signal.entry_price
        sl = signal.stop_loss
        risk_dist = abs(entry - sl)

        if risk_dist <= 0:
            return True

        size_base = (account_equity * self.risk_pct) / risk_dist
        position_size_usd = size_base * entry

        leverage = signal.leverage
        atr = _sf(signal.indicators_snapshot.get("atr_14"))

        if leverage > 0 and atr > 0:
            if signal.direction == "long":
                liq_price = entry * (1 - 1 / leverage + 0.006)
            else:
                liq_price = entry * (1 + 1 / leverage - 0.006)

            if abs(entry - liq_price) < 1.5 * atr:
                return True

        signal.position_size_usd = position_size_usd
        return False

    @staticmethod
    def _rr_gate(signal: SignalEvent) -> bool:
        return signal.risk_reward() < 1.5

    @staticmethod
    def _liquidation_cluster_gate(
        signal: SignalEvent, engine: Any = None,
    ) -> None:
        if engine is None:
            return

        liq_vol = _sf(signal.indicators_snapshot.get("liq_volume_1h"))
        if liq_vol <= 0:
            return

        try:
            with Session(engine) as sess:
                row = sess.execute(
                    sa_text(
                        "SELECT PERCENTILE_CONT(0.99) WITHIN GROUP "
                        "(ORDER BY liq_volume_1h) FROM indicators_5m "
                        "WHERE symbol = :symbol "
                        "AND timestamp > :cutoff"
                    ),
                    {
                        "symbol": signal.symbol,
                        "cutoff": signal.timestamp - timedelta(days=30),
                    },
                ).first()

                if row and row[0] is not None:
                    p99 = float(row[0])
                    if liq_vol > p99 and signal.position_size_usd:
                        signal.position_size_usd *= 0.5
                        signal.regime["liq_cluster_reduced"] = True
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Blocked signal bookkeeping
    # ------------------------------------------------------------------

    def _record_blocked(self, signal: SignalEvent, exit_reason: str) -> None:
        self._blocked.append({
            "signal": signal,
            "exit_reason": exit_reason,
        })
        log.info(
            "signal_blocked",
            gate=exit_reason,
            symbol=signal.symbol,
            direction=signal.direction,
            strategy=signal.strategy_combo,
        )

    @property
    def blocked_count(self) -> int:
        return len(self._blocked)

    def log_blocked_signals(self, engine: Any) -> int:
        """Flush collected blocked signals to trades_log. Returns count."""
        if not self._blocked or engine is None:
            return 0

        count = 0
        with Session(engine) as sess:
            for item in self._blocked:
                sig: SignalEvent = item["signal"]
                record = TradesLog(
                    symbol=sig.symbol,
                    direction=sig.direction,
                    leverage=sig.leverage,
                    entry_time=sig.timestamp,
                    exit_time=sig.timestamp,
                    entry_price=sig.entry_price,
                    exit_price=sig.entry_price,
                    stop_loss=sig.stop_loss,
                    take_profit=sig.take_profit,
                    position_size_usd=0,
                    pnl_pct=0,
                    pnl_usd=0,
                    funding_paid_usd=0,
                    win_loss=None,
                    strategy_combo=sig.strategy_combo,
                    indicators_snapshot=sig.indicators_snapshot,
                    regime_volatility=sig.regime.get("volatility"),
                    regime_funding=sig.regime.get("funding"),
                    regime_time_of_day=sig.regime.get("time_of_day"),
                    exit_reason=item["exit_reason"],
                    is_backtest=self.is_backtest,
                    notes=f"Blocked by {item['exit_reason']}",
                )
                sess.add(record)
                count += 1
            sess.commit()

        log.info("blocked_signals_logged", count=count)
        self._blocked.clear()
        return count


class RiskManagedWrapper:
    """
    Wraps a simulator-compatible strategy (``name`` + ``on_bar()``) with
    RiskManager evaluation so that the existing Simulator can run
    unchanged while signals flow through the risk layer.
    """

    def __init__(
        self,
        inner,
        risk_manager: RiskManager,
        engine: Any,
        symbol: str,
        equity: float = 10_000.0,
        leverage: int = 10,
    ):
        self.inner = inner
        self.rm = risk_manager
        self.engine = engine
        self.name = inner.name
        self.symbol = symbol
        self.equity = equity
        self.leverage = leverage

    def on_bar(
        self, bar: pd.Series, prev_bar: pd.Series | None,
    ):
        from src.backtest.simulator import EntryOrder

        order = self.inner.on_bar(bar, prev_bar)
        if order is None:
            return None

        close = float(bar.get("close", 0))
        ts = pd.Timestamp(bar.get("timestamp"))
        dt = ts.to_pydatetime()

        if order.direction == "long":
            sl = close - order.sl_distance
            tp = close + order.tp_distance
        else:
            sl = close + order.sl_distance
            tp = close - order.tp_distance

        atr_rank = _sf(order.indicators_snapshot.get("atr_pct_rank"))
        funding = _sf(order.indicators_snapshot.get("funding_8h"))

        hour = dt.hour
        if hour < 8:
            session_name = "asia"
        elif hour < 16:
            session_name = "london"
        elif hour < 22:
            session_name = "new_york"
        else:
            session_name = "off_hours"

        if atr_rank < 0.33:
            vol_regime = "low"
        elif atr_rank < 0.66:
            vol_regime = "medium"
        else:
            vol_regime = "high"

        if funding < -0.0001:
            fund_regime = "negative"
        elif funding > 0.0001:
            fund_regime = "positive"
        else:
            fund_regime = "neutral"

        regime = {
            "volatility": vol_regime,
            "funding": fund_regime,
            "time_of_day": session_name,
        }

        signal = SignalEvent(
            symbol=self.symbol,
            direction=order.direction,
            confidence=0.5,
            entry_price=close,
            stop_loss=sl,
            take_profit=tp,
            leverage=self.leverage,
            indicators_snapshot=order.indicators_snapshot,
            strategy_combo=order.strategy_combo,
            regime=regime,
            timestamp=dt,
        )

        funding_rate = _sf(bar.get("funding_8h"))

        result = self.rm.evaluate(
            signal=signal,
            account_equity=self.equity,
            open_positions=[],
            daily_pnl_usd=0.0,
            current_funding_rate=funding_rate,
            predicted_funding_rate=funding_rate,
            session=self.engine,
        )

        if result is None:
            return None

        return EntryOrder(
            direction=result.direction,
            sl_distance=abs(result.entry_price - result.stop_loss),
            tp_distance=abs(result.take_profit - result.entry_price),
            strategy_combo=result.strategy_combo,
            indicators_snapshot=result.indicators_snapshot,
        )
