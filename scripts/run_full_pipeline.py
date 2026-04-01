"""
Master pipeline: download Binance data → compute indicators → run backtests.

Usage:
    python scripts/run_full_pipeline.py
    python scripts/run_full_pipeline.py --symbols BTCUSDT --from 2023-01-01
    python scripts/run_full_pipeline.py --skip-download  # indicators + backtest only
    python scripts/run_full_pipeline.py --skip-indicators  # backtest only
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config import settings  # noqa: E402

# RSI divergence has shown the strongest edge in backtests; momentum strategies use market fills.
STRATEGIES = [
    "rsi_divergence",
    "vwap_reversion",
    "multitf_scalp",
    "bb_squeeze",
    "volume_delta_liq",
    "regime_adaptive",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Full data pipeline")
    parser.add_argument(
        "--symbols", nargs="+", default=settings.symbols,
    )
    parser.add_argument(
        "--from", dest="from_date", default="2021-01-01",
    )
    parser.add_argument(
        "--to", dest="to_date",
        default=(date.today() - timedelta(days=1)).isoformat(),
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-indicators", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--leverage", type=int, default=10)
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument(
        "--strategies", nargs="+", default=STRATEGIES,
        choices=STRATEGIES,
    )
    return parser.parse_args()


def step_download(symbols, from_date, to_date):
    print("\n" + "=" * 70)
    print("STEP 1: Download historical data from Binance")
    print("=" * 70)

    from src.data.binance_downloader import download_klines, download_funding
    from sqlalchemy import create_engine

    engine = create_engine(settings.sync_db_url)

    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        t0 = time.time()
        kline_rows = download_klines(engine, symbol, from_date, to_date)
        print(f"  5m klines: {kline_rows:,} rows ({time.time()-t0:.0f}s)")

        t0 = time.time()
        funding_rows = download_funding(engine, symbol, from_date, to_date)
        print(f"  Funding:   {funding_rows:,} rows ({time.time()-t0:.0f}s)")

    engine.dispose()


def step_indicators(symbols, from_date):
    print("\n" + "=" * 70)
    print("STEP 2: Compute indicators (5m + 15m)")
    print("=" * 70)

    from src.indicators.compute_all import run_pipeline
    from sqlalchemy import create_engine

    engine = create_engine(settings.sync_db_url)

    for symbol in symbols:
        print(f"\n--- {symbol} ---")
        t0 = time.time()
        run_pipeline(engine, symbol, from_date)
        print(f"  Done ({time.time()-t0:.0f}s)")

    engine.dispose()


def step_backtest(symbols, strategies, from_date, to_date, leverage, equity):
    print("\n" + "=" * 70)
    print("STEP 3: Run backtests")
    print("=" * 70)

    from sqlalchemy import create_engine
    from src.backtest.simulator import Simulator
    from src.backtest.shadow_db_logger import ShadowDBLogger
    from src.strategies import STRATEGY_REGISTRY, StrategyAdapter
    import numpy as np
    import math

    engine = create_engine(settings.sync_db_url)
    logger = ShadowDBLogger(engine)

    results = []

    for symbol in symbols:
        for strat_name in strategies:
            print(f"\n  {strat_name} / {symbol} ... ", end="", flush=True)
            t0 = time.time()

            try:
                if strat_name in STRATEGY_REGISTRY:
                    base_cls = STRATEGY_REGISTRY[strat_name]
                    if strat_name == "regime_adaptive":
                        base = base_cls(engine=engine)
                    else:
                        base = base_cls()
                    strategy = StrategyAdapter(
                        base, engine, symbol, from_date, to_date,
                    )
                else:
                    print(f"SKIP (unknown)")
                    continue

                sim = Simulator(
                    strategy=strategy,
                    symbol=symbol,
                    leverage=leverage,
                    risk_pct=0.02,
                    equity=equity,
                )

                trades = sim.run(engine, from_date, to_date)

                combo = trades[0].strategy_combo if trades else [strat_name]
                logger.clear_backtest_trades(
                    strategy_combo=combo,
                    symbol=symbol,
                    from_date=from_date,
                    to_date=to_date,
                )
                logger.log_trades(trades)

                total = len(trades)
                if total > 0:
                    wins = sum(1 for t in trades if t.win_loss)
                    win_rate = wins / total
                    pnl_pcts = [t.pnl_pct for t in trades]
                    avg_pnl = float(np.mean(pnl_pcts))

                    winning = [p for p in pnl_pcts if p > 0]
                    losing = [p for p in pnl_pcts if p <= 0]
                    avg_win = float(np.mean(winning)) if winning else 0.0
                    avg_loss = abs(float(np.mean(losing))) if losing else 0.0
                    loss_rate = 1 - win_rate
                    expectancy = win_rate * avg_win - loss_rate * avg_loss

                    cumulative = np.cumsum(pnl_pcts)
                    running_max = np.maximum.accumulate(cumulative)
                    max_dd = float((cumulative - running_max).min())

                    elapsed = time.time() - t0
                    print(
                        f"{total} trades | WR {win_rate:.1%} | "
                        f"E[r] {expectancy:+.2%} | "
                        f"DD {max_dd:+.2%} ({elapsed:.0f}s)"
                    )

                    results.append({
                        "strategy": strat_name,
                        "symbol": symbol,
                        "trades": total,
                        "win_rate": win_rate,
                        "expectancy": expectancy,
                        "max_dd": max_dd,
                    })
                else:
                    print(f"0 trades ({time.time()-t0:.0f}s)")

            except Exception as exc:
                print(f"ERROR: {exc}")

    engine.dispose()

    if results:
        print("\n" + "=" * 70)
        print("BACKTEST SUMMARY")
        print("=" * 70)
        print(f"{'Strategy':<22} {'Symbol':<10} {'Trades':>7} "
              f"{'WinRate':>8} {'Expect':>8} {'MaxDD':>8}")
        print("-" * 70)
        for r in sorted(results, key=lambda x: x["expectancy"], reverse=True):
            print(
                f"{r['strategy']:<22} {r['symbol']:<10} {r['trades']:>7} "
                f"{r['win_rate']:>7.1%} {r['expectancy']:>+7.2%} "
                f"{r['max_dd']:>+7.2%}"
            )


def main():
    args = parse_args()

    print("=" * 70)
    print("BYBIT FUTURES BOT — Full Pipeline")
    print(f"Symbols:    {args.symbols}")
    print(f"Period:     {args.from_date} -> {args.to_date}")
    print(f"Strategies: {args.strategies}")
    print("=" * 70)

    if not args.skip_download:
        step_download(args.symbols, args.from_date, args.to_date)

    if not args.skip_indicators:
        step_indicators(args.symbols, args.from_date)

    if not args.skip_backtest:
        step_backtest(
            args.symbols, args.strategies,
            args.from_date, args.to_date,
            args.leverage, args.equity,
        )

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print("Next: open the dashboard to explore results:")
    print("  python -m streamlit run src/analytics/streamlit_dashboard.py")


if __name__ == "__main__":
    main()
