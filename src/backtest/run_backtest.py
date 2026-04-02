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
from src.backtest.simulator import ClosedTrade, Simulator  # noqa: E402
from src.backtest.shadow_db_logger import ShadowDBLogger  # noqa: E402
from src.strategies import STRATEGY_REGISTRY, StrategyAdapter  # noqa: E402
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf  # noqa: E402
from src.indicators.compute_all import load_candles  # noqa: E402

log = structlog.get_logger()

ALL_STRATEGY_NAMES = sorted(STRATEGY_REGISTRY.keys())


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
        "--fixed-risk", action="store_true",
        help="Use fixed dollar risk (from initial equity, not current)",
    )
    parser.add_argument(
        "--timeframe", default="5m",
        choices=["5m", "15m", "30m", "1h", "2h", "4h"],
        help="Trading timeframe (default: 5m)",
    )
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


def compute_pnl_dollar_summary(
    trades: list[ClosedTrade],
    initial_equity: float,
) -> dict[str, float | int]:
    """
    Dollar-based metrics for comparing runs (total PnL > win rate).
    """
    pnls = [float(t.pnl_usd) for t in trades]
    n = len(pnls)
    total_pnl = float(np.sum(pnls))
    wins_usd = [p for p in pnls if p > 0]
    losses_usd = [p for p in pnls if p < 0]

    gross_profit = float(np.sum(wins_usd)) if wins_usd else 0.0
    gross_loss = float(np.sum(losses_usd)) if losses_usd else 0.0
    gl_abs = abs(gross_loss)
    if gl_abs > 1e-9:
        profit_factor = gross_profit / gl_abs
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    avg_win_usd = float(np.mean(wins_usd)) if wins_usd else 0.0
    avg_loss_usd = float(np.mean(losses_usd)) if losses_usd else 0.0
    al_abs = abs(avg_loss_usd)
    if al_abs > 1e-9:
        payoff_ratio = avg_win_usd / al_abs
    elif avg_win_usd > 0 and not losses_usd:
        payoff_ratio = float("inf")
    else:
        payoff_ratio = 0.0

    expectancy_usd = float(np.mean(pnls)) if pnls else 0.0

    eq = float(initial_equity)
    peak = eq
    max_dd_usd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = eq - peak
        if dd < max_dd_usd:
            max_dd_usd = dd
    max_dd_usd = abs(max_dd_usd)
    max_dd_vs_peak_pct = (max_dd_usd / peak) if peak > 1e-9 else 0.0

    fees = sum(float(t.fees_paid_usd) for t in trades)
    funding = sum(float(t.funding_paid_usd) for t in trades)

    return {
        "n": n,
        "n_wins": len(wins_usd),
        "n_losses": len(losses_usd),
        "total_pnl_usd": total_pnl,
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "profit_factor": profit_factor,
        "expectancy_usd": expectancy_usd,
        "avg_win_usd": avg_win_usd,
        "avg_loss_usd": avg_loss_usd,
        "payoff_ratio": payoff_ratio,
        "max_dd_usd": max_dd_usd,
        "max_dd_vs_peak_pct": max_dd_vs_peak_pct,
        "fees_paid_usd": fees,
        "funding_net_usd": funding,
        "final_equity": initial_equity + total_pnl,
    }


def _fmt_money(x: float) -> str:
    sign = "+" if x > 0 else ""
    return f"{sign}{x:,.2f}"


def _fmt_pf(x: float) -> str:
    if math.isinf(x):
        return "inf"
    return f"{x:.2f}"


def main() -> None:
    args = parse_args()
    engine = create_engine(settings.sync_db_url)

    base_cls = STRATEGY_REGISTRY[args.strategy]
    if args.strategy == "regime_adaptive":
        base = base_cls(engine=engine)
    else:
        base = base_cls()

    tf = args.timeframe
    trading_bars = None
    context_df = None

    if tf != "5m":
        log.info("building_tf_bars", timeframe=tf, symbol=args.symbol)
        candles_5m = load_candles(engine, args.symbol, args.from_date)
        trading_bars = build_bars_for_tf(candles_5m, tf)
        ctx_tf = CONTEXT_TF.get(tf, tf)
        if ctx_tf != tf:
            context_df = build_bars_for_tf(candles_5m, ctx_tf)
        else:
            context_df = trading_bars
        log.info("tf_bars_ready", tf=tf, bars=len(trading_bars),
                 ctx_tf=ctx_tf, ctx_bars=len(context_df))
    else:
        ctx_tf = CONTEXT_TF.get(tf, "15m")
        if ctx_tf != "15m":
            candles_5m = load_candles(engine, args.symbol, args.from_date)
            context_df = build_bars_for_tf(candles_5m, ctx_tf)

    strategy = StrategyAdapter(
        base, engine, args.symbol, args.from_date, args.to_date,
        context_df=context_df,
    )

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
        fixed_risk=args.fixed_risk,
    )

    trades = sim.run(engine, args.from_date, args.to_date, bars=trading_bars)

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
    max_dd_pct_pts = compute_max_drawdown(pnl_pcts)
    usd = compute_pnl_dollar_summary(trades, args.equity)

    print(f"\nStrategy:     {args.strategy}")
    print(f"Symbol:       {args.symbol}")
    print(f"Timeframe:    {args.timeframe}")
    print(f"Period:       {args.from_date} -> {args.to_date}")
    print(f"Total trades: {total}")
    print()
    print("--- PnL (primary) ---")
    print(f"Start equity:     {_fmt_money(args.equity)} USD")
    print(f"Total PnL:        {_fmt_money(usd['total_pnl_usd'])} USD")
    ret = usd["total_pnl_usd"] / args.equity if args.equity else 0.0
    print(f"Return on start:  {ret:+.2%}")
    print(f"Final equity:     {_fmt_money(usd['final_equity'])} USD")
    print(f"Expectancy/trade: {_fmt_money(usd['expectancy_usd'])} USD")
    print(f"Profit factor:    {_fmt_pf(float(usd['profit_factor']))}")
    print(
        f"Avg win / loss:   {_fmt_money(usd['avg_win_usd'])} / "
        f"{_fmt_money(usd['avg_loss_usd'])} USD  "
        f"(payoff {_fmt_pf(float(usd['payoff_ratio']))}:1)",
    )
    print(f"Gross profit:     {_fmt_money(usd['gross_profit_usd'])} USD")
    print(f"Gross loss:       {_fmt_money(usd['gross_loss_usd'])} USD")
    print(f"Max drawdown:     {usd['max_dd_usd']:,.2f} USD  "
          f"({usd['max_dd_vs_peak_pct']:.2%} vs peak equity)")
    print(
        f"Costs (approx):   fees {_fmt_money(-usd['fees_paid_usd'])}  "
        f"funding {_fmt_money(usd['funding_net_usd'])} USD",
    )
    print()
    print("--- Secondary ---")
    print(f"Win rate:         {win_rate:.1%}  "
          f"({usd['n_wins']} W / {usd['n_losses']} L)")
    print(f"Avg PnL (pct):    {avg_pnl:+.4%}")
    print(f"Expectancy (pct): {expectancy:+.4%}")
    print(f"Sharpe (approx):  {sharpe:.2f}")
    print(f"Max DD (cum %):   {max_dd_pct_pts:+.2%}  "
          "(sum of per-trade pnl_pct; see USD DD above)")
    if blocked_count:
        print(f"Blocked:          {blocked_count} signals")
    print("Trades logged to trades_log")


if __name__ == "__main__":
    main()
