"""
SOL Foundation Test -- year-by-year + walk-forward validation.

Runs the sniper strategy on SOLUSDT across individual years and
split periods to confirm the edge is real, consistent, and not overfit.

Usage
-----
    python scripts/sol_foundation.py
"""

from __future__ import annotations

import math
import sys
import time
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

SYMBOL = "SOLUSDT"
EQUITY = 1_000.0
RISK_PCT = 0.02
FIXED_RISK = True
LEVERAGE = 10

PERIODS = [
    ("2021-01-01", "2022-01-01", "2021 (bull)"),
    ("2022-01-01", "2023-01-01", "2022 (bear)"),
    ("2023-01-01", "2024-01-01", "2023 (recovery)"),
    ("2024-01-01", "2025-01-01", "2024"),
    ("2025-01-01", "2026-04-01", "2025-26 (recent)"),
    ("2021-01-01", "2023-07-01", "IN-SAMPLE (21-mid23)"),
    ("2023-07-01", "2026-04-01", "OUT-OF-SAMPLE (mid23-26)"),
    ("2021-01-01", "2026-04-01", "FULL 5yr"),
]

TIMEFRAMES = ["15m", "4h"]


def run_one(engine, candles_5m, tf, from_d, to_d) -> dict | None:
    built_cache = {}
    needed = {tf}
    ctx_tf = CONTEXT_TF.get(tf, tf)
    needed.add(ctx_tf)

    for t in needed:
        if t != "5m" and t not in built_cache:
            built_cache[t] = build_bars_for_tf(candles_5m, t)

    trading_bars = built_cache.get(tf)
    context_df = built_cache.get(ctx_tf, trading_bars)

    # Filter trading bars to period
    if trading_bars is not None:
        mask = (trading_bars["timestamp"] >= from_d) & (
            trading_bars["timestamp"] < to_d
        )
        trading_bars = trading_bars.loc[mask].reset_index(drop=True)
        if trading_bars.empty:
            return None

    base = SniperStrategy()
    adapter = StrategyAdapter(
        base, engine, SYMBOL, from_d, to_d, context_df=context_df,
    )
    sim = Simulator(
        strategy=adapter,
        symbol=SYMBOL,
        leverage=LEVERAGE,
        risk_pct=RISK_PCT,
        equity=EQUITY,
        fixed_risk=FIXED_RISK,
    )
    trades = sim.run(engine, from_d, to_d, bars=trading_bars)

    if not trades:
        return {"trades": 0, "wr": 0, "payoff": 0, "pnl": 0, "ret": 0,
                "sharpe": 0, "max_dd": 0, "pf": 0}

    usd = compute_pnl_dollar_summary(trades, EQUITY)
    pnl_pcts = [t.pnl_pct for t in trades]
    total = len(trades)
    wins = sum(1 for t in trades if t.win_loss)

    from datetime import datetime as dt
    d0, d1 = dt.fromisoformat(from_d), dt.fromisoformat(to_d)
    days = max((d1 - d0).days, 1)
    tpy = total / days * 365
    sharpe = compute_sharpe(pnl_pcts, tpy)

    return {
        "trades": total,
        "wr": wins / total if total else 0,
        "payoff": usd["payoff_ratio"],
        "pnl": usd["total_pnl_usd"],
        "ret": usd["total_pnl_usd"] / EQUITY,
        "sharpe": sharpe,
        "max_dd": usd["max_dd_vs_peak_pct"],
        "pf": usd["profit_factor"],
    }


def fmt_pf(x):
    return "inf" if math.isinf(x) else f"{x:.2f}"


def main():
    engine = create_engine(settings.sync_db_url)
    print(f"\nLoading {SYMBOL} 5m candles ...", flush=True)
    t0 = time.time()
    candles_5m = load_candles(engine, SYMBOL)
    print(f"Loaded {len(candles_5m):,} bars ({time.time()-t0:.1f}s)\n",
          flush=True)

    for tf in TIMEFRAMES:
        print(f"{'='*78}")
        print(f"  SOLUSDT / {tf}  --  ${EQUITY:,.0f} equity, "
              f"{RISK_PCT:.0%} risk, {'fixed' if FIXED_RISK else 'pct'} sizing")
        print(f"{'='*78}")
        hdr = (f"{'Period':<25} {'Trades':>6}  {'WR':>6}  {'Payoff':>7}  "
               f"{'PnL $':>10}  {'Return':>8}  {'Sharpe':>6}  "
               f"{'MaxDD':>6}  {'PF':>5}")
        print(hdr)
        print("-" * len(hdr))

        for from_d, to_d, label in PERIODS:
            t1 = time.time()
            r = run_one(engine, candles_5m, tf, from_d, to_d)
            el = time.time() - t1

            if r is None or r["trades"] == 0:
                print(f"{label:<25} {'0':>6}  {'--':>6}  {'--':>7}  "
                      f"{'$0':>10}  {'--':>8}  {'--':>6}  "
                      f"{'--':>6}  {'--':>5}  ({el:.0f}s)")
                continue

            print(
                f"{label:<25} {r['trades']:>6}  {r['wr']:>5.1%}  "
                f"{r['payoff']:>6.2f}x  {r['pnl']:>+10,.2f}  "
                f"{r['ret']:>+7.1%}  {r['sharpe']:>6.2f}  "
                f"{r['max_dd']:>5.1%}  {fmt_pf(r['pf']):>5}  "
                f"({el:.0f}s)"
            )
        print()


if __name__ == "__main__":
    main()
