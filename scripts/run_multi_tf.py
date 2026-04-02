"""
Run the sniper strategy across multiple symbols x timeframes.

Loads 5m candles once per symbol, then resamples to each target TF,
computes indicators, and runs the backtest.  Prints a summary table.

Usage
-----
    python scripts/run_multi_tf.py --from 2021-01-01 --to 2026-03-31 \
        --equity 1000 --risk-pct 0.05 --fixed-risk
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import structlog
from sqlalchemy import create_engine

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config import settings  # noqa: E402
from src.backtest.simulator import Simulator  # noqa: E402
from src.backtest.run_backtest import (  # noqa: E402
    compute_pnl_dollar_summary,
    compute_sharpe,
)
from src.indicators.compute_all import load_candles  # noqa: E402
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf  # noqa: E402
from src.strategies import StrategyAdapter  # noqa: E402
from src.strategies.strategy_sniper import SniperStrategy  # noqa: E402

log = structlog.get_logger()

SYMBOLS = ["ETHUSDT", "BTCUSDT", "SOLUSDT"]
TIMEFRAMES = ["15m", "1h", "4h"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-TF sniper backtest")
    p.add_argument("--from", dest="from_date", default="2021-01-01")
    p.add_argument(
        "--to", dest="to_date",
        default=(date.today() - timedelta(days=1)).isoformat(),
    )
    p.add_argument("--equity", type=float, default=1_000.0)
    p.add_argument("--risk-pct", type=float, default=0.05)
    p.add_argument("--fixed-risk", action="store_true")
    p.add_argument("--leverage", type=int, default=10)
    p.add_argument("--symbols", nargs="+", default=SYMBOLS)
    p.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    return p.parse_args()


def run_combo(
    engine,
    symbol: str,
    tf: str,
    trading_bars,
    context_df,
    args,
) -> dict | None:
    base = SniperStrategy()
    adapter = StrategyAdapter(
        base, engine, symbol, args.from_date, args.to_date,
        context_df=context_df,
    )
    sim = Simulator(
        strategy=adapter,
        symbol=symbol,
        leverage=args.leverage,
        risk_pct=args.risk_pct,
        equity=args.equity,
        fixed_risk=args.fixed_risk,
    )
    trades = sim.run(engine, args.from_date, args.to_date, bars=trading_bars)
    if not trades:
        return None

    usd = compute_pnl_dollar_summary(trades, args.equity)
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = sum(1 for t in trades if t.win_loss)
    total = len(trades)
    wr = wins / total if total else 0.0

    from datetime import datetime as dt
    d0 = dt.fromisoformat(args.from_date)
    d1 = dt.fromisoformat(args.to_date)
    days = max((d1 - d0).days, 1)
    trades_per_year = total / days * 365
    sharpe = compute_sharpe(pnl_pcts, trades_per_year)

    return {
        "symbol": symbol,
        "tf": tf,
        "trades": total,
        "wr": wr,
        "payoff": usd["payoff_ratio"],
        "pnl": usd["total_pnl_usd"],
        "ret": usd["total_pnl_usd"] / args.equity if args.equity else 0,
        "sharpe": sharpe,
        "max_dd_pct": usd["max_dd_vs_peak_pct"],
        "pf": usd["profit_factor"],
    }


def main() -> None:
    args = parse_args()
    engine = create_engine(settings.sync_db_url)
    results: list[dict] = []

    for symbol in args.symbols:
        t0 = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"  Loading {symbol} ...", flush=True)
        candles_5m = load_candles(engine, symbol, args.from_date)
        print(f"  Loaded {len(candles_5m):,} 5m candles "
              f"({time.time()-t0:.1f}s)", flush=True)

        built: dict[str, object] = {}
        needed_tfs = set(args.timeframes)
        for tf in args.timeframes:
            ctx = CONTEXT_TF.get(tf, tf)
            needed_tfs.add(ctx)

        for tf in sorted(needed_tfs):
            if tf == "5m":
                continue
            t1 = time.time()
            built[tf] = build_bars_for_tf(candles_5m, tf)
            n = len(built[tf])
            print(f"  Built {tf:>3s} bars: {n:>7,}  ({time.time()-t1:.1f}s)",
                  flush=True)

        for tf in args.timeframes:
            trading_bars = built.get(tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            context_df = built.get(ctx_tf, trading_bars)

            print(f"  Running {symbol} / {tf} ...", end=" ", flush=True)
            t1 = time.time()
            res = run_combo(
                engine, symbol, tf, trading_bars, context_df, args,
            )
            elapsed = time.time() - t1
            if res:
                results.append(res)
                print(f"{res['trades']:>4} trades  "
                      f"WR {res['wr']:5.1%}  "
                      f"PnL {res['pnl']:>+10,.2f}  "
                      f"({elapsed:.1f}s)", flush=True)
            else:
                print(f"   0 trades  ({elapsed:.1f}s)", flush=True)

    # ── Summary table ──────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  SNIPER STRATEGY -- MULTI-TF BACKTEST RESULTS")
    print(f"  {args.from_date} -> {args.to_date}  |  "
          f"equity ${args.equity:,.0f}  |  risk {args.risk_pct:.0%}  |  "
          f"{'fixed' if args.fixed_risk else 'pct'} sizing")
    print(f"{'='*80}")
    hdr = (f"{'Symbol':<10} {'TF':>4}  {'Trades':>6}  {'WR':>6}  "
           f"{'Payoff':>7}  {'PnL $':>10}  {'Return':>8}  "
           f"{'Sharpe':>6}  {'MaxDD':>7}  {'PF':>5}")
    print(hdr)
    print("-" * len(hdr))

    for r in results:
        pf_str = f"{r['pf']:.2f}" if not math.isinf(r["pf"]) else "inf"
        print(
            f"{r['symbol']:<10} {r['tf']:>4}  {r['trades']:>6}  "
            f"{r['wr']:>5.1%}  {r['payoff']:>6.2f}x  "
            f"{r['pnl']:>+10,.2f}  {r['ret']:>+7.1%}  "
            f"{r['sharpe']:>6.2f}  {r['max_dd_pct']:>6.1%}  "
            f"{pf_str:>5}"
        )

    if results:
        total_pnl = sum(r["pnl"] for r in results)
        total_capital = args.equity * len(results)
        total_trades = sum(r["trades"] for r in results)
        avg_wr_weighted = (
            sum(r["wr"] * r["trades"] for r in results) / total_trades
            if total_trades else 0
        )
        print("-" * len(hdr))
        print(
            f"{'TOTAL':<10} {'':>4}  {total_trades:>6}  "
            f"{avg_wr_weighted:>5.1%}  {'':>7}  "
            f"{total_pnl:>+10,.2f}  "
            f"{total_pnl/total_capital:>+7.1%}  "
            f"{'':>6}  {'':>7}  {'':>5}"
        )
        print(f"\nDiversified: {len(results)} combos x "
              f"${args.equity:,.0f} = ${total_capital:,.0f} capital  ->  "
              f"${total_capital + total_pnl:,.2f} final")


if __name__ == "__main__":
    main()
