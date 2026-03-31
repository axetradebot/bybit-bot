"""
CLI entry point: run a backtest strategy over a date range.

Usage
-----
    python src/backtest/run_backtest.py \
        --strategy bb_squeeze \
        --symbol BTCUSDT \
        --from 2024-01-01 \
        --to 2024-01-02 \
        --leverage 10 \
        --risk-pct 0.02
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import structlog
from sqlalchemy import create_engine

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import settings  # noqa: E402
from src.backtest.simulator import Simulator, STRATEGIES  # noqa: E402
from src.backtest.shadow_db_logger import ShadowDBLogger  # noqa: E402
from src.strategies import STRATEGY_REGISTRY, StrategyAdapter  # noqa: E402

log = structlog.get_logger()

ALL_STRATEGY_NAMES = sorted(set(list(STRATEGIES.keys()) + list(STRATEGY_REGISTRY.keys())))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a backtest strategy over a date range",
    )
    parser.add_argument(
        "--strategy", required=True,
        choices=ALL_STRATEGY_NAMES,
        help="Strategy name",
    )
    parser.add_argument("--symbol", required=True, help="e.g. BTCUSDT")
    parser.add_argument(
        "--from", dest="from_date", default="2021-01-01",
        help="Start date inclusive (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--to", dest="to_date",
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="End date non-inclusive (YYYY-MM-DD)",
    )
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--risk-pct", type=float, default=0.02)
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument(
        "--with-risk-manager", action="store_true",
        help="Run signals through the RiskManager gate layer",
    )
    return parser.parse_args()


def compute_sharpe(pnl_pcts: list[float], trades_per_year: float) -> float:
    if len(pnl_pcts) < 2:
        return 0.0
    arr = np.array(pnl_pcts)
    std = arr.std(ddof=1)
    if std == 0:
        return 0.0
    return float(arr.mean() / std * math.sqrt(trades_per_year))


def compute_max_drawdown(pnl_pcts: list[float]) -> float:
    if not pnl_pcts:
        return 0.0
    cumulative = np.cumsum(pnl_pcts)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    return float(drawdowns.min()) if len(drawdowns) > 0 else 0.0


def main() -> None:
    args = parse_args()
    engine = create_engine(settings.sync_db_url)

    if args.strategy in STRATEGY_REGISTRY:
        base_cls = STRATEGY_REGISTRY[args.strategy]
        if args.strategy == "regime_adaptive":
            base = base_cls(engine=engine)
        else:
            base = base_cls()
        strategy = StrategyAdapter(
            base, engine, args.symbol, args.from_date, args.to_date,
        )
    else:
        strategy_cls = STRATEGIES[args.strategy]
        strategy = strategy_cls()

    rm = None
    if args.with_risk_manager:
        from src.risk.risk_manager import RiskManager, RiskManagedWrapper
        rm = RiskManager(is_backtest=True)
        strategy = RiskManagedWrapper(
            inner=strategy,
            risk_manager=rm,
            engine=engine,
            symbol=args.symbol,
            equity=args.equity,
            leverage=args.leverage,
        )

    sim = Simulator(
        strategy=strategy,
        symbol=args.symbol,
        leverage=args.leverage,
        risk_pct=args.risk_pct,
        equity=args.equity,
    )

    trades = sim.run(engine, args.from_date, args.to_date)

    logger = ShadowDBLogger(engine)
    # Use the actual strategy_combo from the first trade for cleanup;
    # fall back to [strategy.name] if no trades were generated.
    combo = trades[0].strategy_combo if trades else [strategy.name]
    logger.clear_backtest_trades(
        strategy_combo=combo,
        symbol=args.symbol,
        from_date=args.from_date,
        to_date=args.to_date,
    )
    logger.log_trades(trades)

    blocked_count = 0
    if rm is not None:
        blocked_count = rm.log_blocked_signals(engine)

    # --- Summary ---
    total = len(trades)
    if total == 0:
        print(f"\nStrategy:     {args.strategy}")
        print(f"Symbol:       {args.symbol}")
        print(f"Period:       {args.from_date} -> {args.to_date}")
        print("Total trades: 0")
        print("No trades generated.")
        return

    wins = sum(1 for t in trades if t.win_loss)
    win_rate = wins / total
    pnl_pcts = [t.pnl_pct for t in trades]
    avg_pnl = np.mean(pnl_pcts)

    winning = [p for p in pnl_pcts if p > 0]
    losing = [p for p in pnl_pcts if p <= 0]
    avg_win = np.mean(winning) if winning else 0.0
    avg_loss = abs(np.mean(losing)) if losing else 0.0
    loss_rate = 1 - win_rate
    expectancy = win_rate * avg_win - loss_rate * avg_loss

    from datetime import datetime as dt
    d0 = dt.fromisoformat(args.from_date)
    d1 = dt.fromisoformat(args.to_date)
    days = max((d1 - d0).days, 1)
    trades_per_year = total / days * 365

    sharpe = compute_sharpe(pnl_pcts, trades_per_year)
    max_dd = compute_max_drawdown(pnl_pcts)

    print(f"\nStrategy:     {args.strategy}")
    print(f"Symbol:       {args.symbol}")
    print(f"Period:       {args.from_date} -> {args.to_date}")
    print(f"Total trades: {total}")
    print(f"Win rate:     {win_rate:.1%}")
    print(f"Avg PnL:      {avg_pnl:+.2%}")
    print(f"Expectancy:   {expectancy:+.2%}")
    print(f"Sharpe:       {sharpe:.2f}")
    print(f"Max drawdown: {max_dd:+.2%}")
    if blocked_count:
        print(f"Blocked:      {blocked_count} signals")
    print("Trades logged to trades_log")


if __name__ == "__main__":
    main()
