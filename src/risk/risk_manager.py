"""
Risk Manager — hard gate between strategy signals and order execution.

Every signal must pass through ``evaluate()`` before being acted on.
Blocked signals are collected in memory and flushed to ``trades_log``
via ``log_blocked_signals()`` so they remain visible in analytics.

v2 (Apr-2026) additions
-----------------------
- Continuous drawdown throttle (replaces binary 5% daily-loss halt)
- Confidence-scaled fractional Kelly sizing (replaces flat risk_pct)
- Portfolio β-BTC notional cap (replaces naive max-3-positions)
- Volatility-targeting equity overlay
- Bayesian regime filter (replaces hard-coded blocked_regimes)
- Strategy auto-disable on rolling negative expectancy
- Funding-aware pre-trade expectancy filter
- Funding-fix proximity exit suggestion
- Honours per-signal min_rr / risk_pct (no longer silently overrides)
- _ema_trend_risk now *scales* the configured risk_pct instead of
  overwriting it with hard-coded 0.0075/0.0125
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
from src.risk.portfolio import (
    BayesianRegimeFilter,
    DrawdownThrottle,
    FundingExpectancyFilter,
    KellySizer,
    PortfolioCorrelationGate,
    StrategyAutoDisable,
    VolatilityTargetOverlay,
    minutes_to_next_funding_fix,
)
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

    # _ema_trend_risk is now a *scale* of the base risk_pct rather
    # than a hard override.  Range: aligned trend gets +25%, against-
    # trend gets −30%, neutral leaves it untouched.
    TREND_BOOST_MULT = 1.25
    TREND_REDUCE_MULT = 0.70
    EMA_NEUTRAL_BAND = 0.001

    # Hard-cap floor/ceiling applied after every other multiplier.
    RISK_FLOOR_PCT = 0.0025
    RISK_CEILING_PCT = 0.030

    # Block trades opened within FUNDING_PROXIMITY_MIN of a funding
    # fix when funding is severely against the trade's direction.
    FUNDING_PROXIMITY_MIN = 30
    FUNDING_PROXIMITY_RATE = 0.0003

    # ── NEW-6 anti-chop floor (best of grid search) ──
    # Production filter that came out of best_filters_sim_v2:
    # DI-alignment + on the right side of BB mid + volume_ratio >= 1.2
    # + no BB squeeze.  Strategies in this set have it applied as a hard
    # gate; strategies in CHOP_FILTER_EXEMPT (mean-reversion / squeeze
    # plays) thrive in the conditions NEW-6 rejects.
    CHOP_VOL_FLOOR = 1.2
    CHOP_FILTER_STRATEGIES = frozenset({
        "sniper", "multitf_scalp", "high_winrate", "volume_delta_liq",
    })

    def __init__(
        self,
        is_backtest: bool = True,
        risk_pct: float = 0.005,  # validated by portfolio_sim_v3 (+12% / 48% DD)
        engine: Any | None = None,
        max_concurrent_positions: int = 3,
    ):
        self.is_backtest = is_backtest
        self.risk_pct = risk_pct
        self.max_concurrent_positions = max_concurrent_positions

        self._daily_loss_halted = False
        self._halt_date: date | None = None
        self._blocked: list[dict] = []

        # ── Portfolio primitives ──
        self._kelly = KellySizer(engine=engine, default_risk_pct=risk_pct)
        self._dd = DrawdownThrottle()
        self._corr_gate = PortfolioCorrelationGate(engine=engine)
        self._vol_target = VolatilityTargetOverlay(engine=engine)
        self._regime_filter = BayesianRegimeFilter(engine=engine)
        self._auto_disable = StrategyAutoDisable(engine=engine)
        self._funding_filter = FundingExpectancyFilter(engine=engine)

    # ------------------------------------------------------------------
    # Public helpers used by the live listener
    # ------------------------------------------------------------------

    def is_strategy_disabled(self, strategy_name: str) -> bool:
        return self._auto_disable.is_disabled(strategy_name)

    @property
    def disabled_strategies(self) -> set[str]:
        return self._auto_disable.disabled

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

        Gate order (first block wins):
          1.  auto_disable_gate         (strategy on the kill list?)
          2.  drawdown_throttle_gate    (hard halt at 20% DD)
          3.  daily_loss_gate           (legacy 5% / day, kept as belt+suspenders)
          4.  session_open_gate         (avoid funding-fix windows)
          5.  funding_proximity_gate    (skip trades right before fix when adverse)
          5b. chop_gate                 (NEW-6 anti-chop, trend strategies only)
          6.  bayesian_regime_gate      (data-driven regime block)
          7.  funding_expectancy_gate   (cost > 50% of expected edge?)
          8.  funding_gate              (legacy hard funding cap)
          9.  rr_gate                   (signal.min_rr / default 1.5)
          10. leverage_gate             (modifies leverage, never blocks)
          11. position_size_gate        (Kelly + DD + vol-target multipliers)
          12. portfolio_correlation_gate(downsizes or rejects)
          13. liquidation_cluster_gate  (reduces size if extreme)
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

        # Gate 1 — strategy auto-disable
        head_strategy = (
            approved.strategy_combo[0] if approved.strategy_combo else ""
        )
        if head_strategy and self._auto_disable.is_disabled(head_strategy):
            self._record_blocked(signal, "auto_disabled")
            return None

        # Gate 1b — same-symbol dedup + concurrent position cap
        if self._max_positions_gate(signal.symbol, open_positions):
            self._record_blocked(signal, "max_positions_gate")
            return None

        # Gate 2 — drawdown-aware risk multiplier (also yields hard halt)
        dd_mult, dd_halt = self._dd.multiplier(account_equity)
        if dd_halt:
            self._record_blocked(signal, "drawdown_halt")
            return None

        # Gate 3 — legacy daily loss (now a 5% catastrophic backstop)
        if self._daily_loss_halted:
            self._record_blocked(signal, "daily_loss_gate")
            return None
        if self._daily_loss_gate(account_equity, daily_pnl_usd):
            self._daily_loss_halted = True
            self._halt_date = sig_date
            self._record_blocked(signal, "daily_loss_gate")
            return None

        # Gate 4 — session-open / funding-fix proximity
        if self._session_open_gate(signal.timestamp):
            self._record_blocked(signal, "session_gate")
            return None

        # Gate 5 — adverse funding right before a fix
        if self._funding_proximity_gate(
            signal.timestamp, signal.direction, current_funding_rate,
        ):
            self._record_blocked(signal, "funding_proximity")
            return None

        # Gate 5b — NEW-6 anti-chop filter (trend-strategies only)
        if self._chop_gate(approved):
            self._record_blocked(signal, "chop_filter")
            return None

        # Gate 6 — Bayesian regime filter
        blocked, posterior = self._regime_filter.is_blocked(
            approved.strategy_combo, approved.regime,
        )
        if blocked:
            approved.regime["posterior_expectancy_pct"] = posterior
            self._record_blocked(signal, "regime_filter")
            return None

        # Gate 7 — funding cost vs expected edge
        funding_blocked, funding_cost = self._funding_filter.is_blocked(
            approved.strategy_combo, approved.direction,
            current_funding_rate,
        )
        if funding_blocked:
            approved.regime["funding_cost_pct"] = funding_cost
            self._record_blocked(signal, "funding_expectancy")
            return None

        # Gate 8 — legacy hard-funding gate
        if self._funding_gate(
            signal.direction, current_funding_rate, predicted_funding_rate,
        ):
            self._record_blocked(signal, "funding_gate")
            return None

        # Gate 9 — strategy-aware R:R gate (uses signal.min_rr)
        if self._rr_gate(approved):
            self._record_blocked(signal, "rr_gate")
            return None

        # Gate 10 — leverage cap (never blocks, never raises beyond signal.leverage)
        self._leverage_gate(approved, session)

        # Gate 11 — Kelly + DD + vol-target sizing
        if self._position_size_gate(
            approved, account_equity, dd_mult,
        ):
            self._record_blocked(signal, "liquidation_too_close")
            return None

        # Gate 12 — portfolio β-BTC notional cap
        allow, scaled = self._corr_gate.evaluate(
            candidate_symbol=approved.symbol,
            candidate_notional=approved.position_size_usd or 0.0,
            candidate_direction=approved.direction,
            equity=account_equity,
            open_positions=open_positions,
        )
        if not allow:
            self._record_blocked(signal, "portfolio_beta_cap")
            return None
        approved.position_size_usd = scaled

        # Gate 13 — extreme-liquidation-cluster downsizing
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

    @classmethod
    def _funding_proximity_gate(
        cls,
        ts: datetime,
        direction: str,
        current_funding_rate: float,
    ) -> bool:
        """
        Block trades that would *pay* funding inside the next
        FUNDING_PROXIMITY_MIN minutes when funding is severely adverse.
        """
        if abs(current_funding_rate) < cls.FUNDING_PROXIMITY_RATE:
            return False
        mins_to_fix = minutes_to_next_funding_fix(ts)
        if mins_to_fix > cls.FUNDING_PROXIMITY_MIN:
            return False
        if direction == "long" and current_funding_rate > 0:
            return True
        if direction == "short" and current_funding_rate < 0:
            return True
        return False

    def _chop_gate(self, signal: SignalEvent) -> bool:
        """NEW-6 anti-chop gate. Returns True to BLOCK.

        Identified by best_filters_sim_v2 as the highest-PnL filter in
        a 19-config grid search:  DI alignment + on the right side of
        BB mid + volume_ratio >= 1.2x + no BB squeeze.  Applied only to
        trend-following strategies (CHOP_FILTER_STRATEGIES); mean-rev /
        squeeze plays are exempt because they thrive precisely in the
        regimes this filter rejects.
        """
        head = (signal.strategy_combo or [""])[0]
        if head not in self.CHOP_FILTER_STRATEGIES:
            return False

        snap = signal.indicators_snapshot or {}
        direction = signal.direction

        plus_di = _sf(snap.get("plus_di"))
        minus_di = _sf(snap.get("minus_di"))
        if plus_di > 0 or minus_di > 0:
            if direction == "long" and plus_di <= minus_di:
                return True
            if direction == "short" and minus_di <= plus_di:
                return True

        bb_mid = _sf(snap.get("bb_mid"))
        close = _sf(snap.get("close"))
        if bb_mid > 0 and close > 0:
            if direction == "long" and close <= bb_mid:
                return True
            if direction == "short" and close >= bb_mid:
                return True

        vol_ratio = _sf(snap.get("volume_ratio"))
        if vol_ratio > 0 and vol_ratio < self.CHOP_VOL_FLOOR:
            return True

        if snap.get("bb_squeeze") is True:
            return True

        return False

    def _max_positions_gate(
        self, symbol: str, open_positions: list[dict],
    ) -> bool:
        """Same-symbol dedup is hard; portfolio cap is via correlation gate."""
        if len(open_positions) >= self.max_concurrent_positions:
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
            max_lev = 25
        elif total_trades >= 100 and win_rate >= 0.58:
            max_lev = 20
        else:
            max_lev = 20

        max_lev = min(max_lev, 25)
        signal.leverage = min(signal.leverage, max_lev)

    def _ema_trend_scale(self, signal: SignalEvent) -> float:
        """
        Returns a *multiplier* applied to the base risk_pct based on
        EMA 50 / 200 trend alignment.  Aligned with trend → +25 %;
        against trend → −30 %; neutral band → 1.0 ×.
        """
        ema50 = _sf(signal.indicators_snapshot.get("ema_50"))
        ema200 = _sf(signal.indicators_snapshot.get("ema_200"))
        if ema50 <= 0 or ema200 <= 0:
            return 1.0

        spread = (ema50 - ema200) / ema200
        if abs(spread) <= self.EMA_NEUTRAL_BAND:
            return 1.0

        bullish = spread > 0
        with_trend = ((signal.direction == "long" and bullish) or
                      (signal.direction == "short" and not bullish))
        return self.TREND_BOOST_MULT if with_trend else self.TREND_REDUCE_MULT

    # Kept for back-compat — RiskManagedWrapper / tests may still call it.
    def _ema_trend_risk(self, signal: SignalEvent) -> float:
        return self.risk_pct * self._ema_trend_scale(signal)

    def _resolve_risk_pct(
        self,
        signal: SignalEvent,
        dd_mult: float,
    ) -> float:
        """
        Compose: base_risk × ema_trend_scale × kelly_recommendation_ratio
                 × dd_mult × vol_target_mult, clipped to floor/ceiling.

        Honours an explicit ``signal.risk_pct`` (no Kelly override).
        """
        if signal.risk_pct is not None:
            base = signal.risk_pct
        else:
            base = self._kelly.recommended_risk_pct(
                combo=signal.strategy_combo,
                symbol=signal.symbol,
                confidence=signal.confidence,
                risk_floor=self.RISK_FLOOR_PCT,
                risk_ceiling=self.RISK_CEILING_PCT,
            )

        scaled = (
            base
            * self._ema_trend_scale(signal)
            * dd_mult
            * self._vol_target.multiplier()
        )
        return max(self.RISK_FLOOR_PCT, min(scaled, self.RISK_CEILING_PCT))

    def _position_size_gate(
        self,
        signal: SignalEvent,
        account_equity: float,
        dd_mult: float = 1.0,
    ) -> bool:
        entry = signal.entry_price
        sl = signal.stop_loss
        risk_dist = abs(entry - sl)

        if risk_dist <= 0:
            return True

        MAKER_FEE = 0.0002
        TAKER_FEE = 0.00055
        SLIPPAGE_BPS = 0.0003
        round_trip_cost_rate = MAKER_FEE + TAKER_FEE + SLIPPAGE_BPS

        effective_risk = self._resolve_risk_pct(signal, dd_mult)
        # Stamp the resolved value back so the order manager / sim see it.
        signal.risk_pct = effective_risk

        risk_amount = account_equity * effective_risk
        size_base = risk_amount / (risk_dist + entry * round_trip_cost_rate)
        position_size_usd = size_base * entry

        leverage = signal.leverage or 20
        max_notional = account_equity * leverage * 0.25
        if position_size_usd > max_notional:
            position_size_usd = max_notional

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
        """Strategy-aware: honours signal.min_rr (defaults to 1.5)."""
        return signal.risk_reward() < signal.effective_min_rr(1.5)

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

        fill_mode = getattr(result, "fill_mode", None) or "market"
        return EntryOrder(
            direction=result.direction,
            sl_distance=abs(result.entry_price - result.stop_loss),
            tp_distance=abs(result.take_profit - result.entry_price),
            strategy_combo=result.strategy_combo,
            indicators_snapshot=result.indicators_snapshot,
            limit_price=float(result.entry_price),
            fill_mode=fill_mode,
        )
