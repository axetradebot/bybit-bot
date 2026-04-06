"""
Portfolio simulation: all pairs share ONE equity pool.

Uses ``src/backtest/live_aligned_portfolio.py`` so fees, entry (aggressive
limit + TTL), TP (maker) / SL-trail-timeout (taker), slippage on stops,
and funding match the live bot model documented there.

Usage:
    python scripts/portfolio_sim.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.backtest.live_aligned_portfolio import (
    LivePortfolioTrade,
    collect_trades_live_aligned,
    run_portfolio_live_aligned,
)

PrecomputedTrade = LivePortfolioTrade

LEVERAGE = 20
START_EQUITY = 3000.0
RISK_PCT = 0.02
MAX_CONCURRENT = 4

SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT",
    "1000PEPEUSDT", "DOGEUSDT", "OPUSDT", "SUIUSDT",
]
TIMEFRAMES = ["15m", "4h"]
FROM = "2023-01-01"
TO = "2026-04-03"


def collect_and_resolve_trades(trading_bars, context_df, symbol, tf):
    """Live-aligned collect (limit entry + TTL, absolute SL/TP from signal)."""
    rows = collect_trades_live_aligned(
        trading_bars, context_df, symbol, tf, snap_fn=lambda _b, _c: {},
    )
    return [t for t, _ in rows]


def _monthly_from_curve(
    equity_curve: list, start_equity: float,
) -> dict[str, float]:
    curve_by_month: dict[str, list] = {}
    for ts, eq in equity_curve:
        key = str(ts)[:7]
        if key not in curve_by_month:
            curve_by_month[key] = []
        curve_by_month[key].append(eq)
    monthly: dict[str, float] = {}
    prev_month_end = start_equity
    for m in sorted(curve_by_month.keys()):
        month_end = curve_by_month[m][-1]
        monthly[m] = month_end - prev_month_end
        prev_month_end = month_end
    return monthly


def run_portfolio(all_trades: list[PrecomputedTrade]) -> dict:
    """Shared equity pool; live-aligned fees + funding."""
    r = run_portfolio_live_aligned(
        all_trades,
        start_equity=START_EQUITY,
        risk_pct=RISK_PCT,
        max_concurrent=MAX_CONCURRENT,
        track_equity_curve=True,
        curve_from=FROM,
    )
    eq_curve = r.get("equity_curve", [(pd.Timestamp(FROM), START_EQUITY)])
    r["monthly"] = _monthly_from_curve(eq_curve, START_EQUITY)
    return r


def load_candles(symbol, cache_dir):
    cache_path = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return pd.DataFrame()


def main():
    cache_dir = project_root / "data_cache"

    print("=" * 75, flush=True)
    print("PORTFOLIO SIMULATION: Shared equity, concurrent positions", flush=True)
    print(f"Equity: ${START_EQUITY:,.0f}  |  Leverage: {LEVERAGE}x  |  "
          f"Max concurrent: {MAX_CONCURRENT}", flush=True)
    print(f"Risk levels to compare: {', '.join(f'{r:.1%}' for r in [0.01, 0.015, 0.02])}",
          flush=True)
    print("=" * 75, flush=True)

    # Collect and resolve all trades
    print("\n-- Collecting & resolving trades --", flush=True)
    all_trades: list[PrecomputedTrade] = []

    for sym in SYMBOLS:
        df_5m = load_candles(sym, cache_dir)
        if df_5m.empty:
            continue
        for tf in TIMEFRAMES:
            t0 = time.time()
            context_tf = CONTEXT_TF.get(tf, "4h")
            context_df = build_bars_for_tf(df_5m, context_tf)
            trading_bars = build_bars_for_tf(df_5m, tf)

            trades = collect_and_resolve_trades(
                trading_bars, context_df, sym, tf)
            all_trades.extend(trades)

            elapsed = time.time() - t0
            print(f"  {tf} {sym}: {len(trades)} trades  ({elapsed:.1f}s)",
                  flush=True)

    all_trades.sort(key=lambda t: t.entry_time)
    print(f"\n  Total resolved trades: {len(all_trades)}", flush=True)

    # Run portfolio simulation at multiple risk levels
    risk_levels = [0.01, 0.015, 0.02]
    results_by_risk: dict[float, dict] = {}

    for rp in risk_levels:
        global RISK_PCT
        RISK_PCT = rp
        print(f"\n-- Running portfolio at {rp:.1%} risk --", flush=True)
        r = run_portfolio(all_trades)
        results_by_risk[rp] = r

    # Comparison table
    print("\n" + "=" * 75, flush=True)
    print("PORTFOLIO COMPARISON", flush=True)
    print("=" * 75, flush=True)
    print(f"{'Risk':>6s}  {'Final Eq':>10s}  {'PnL':>12s}  {'Return':>8s}  "
          f"{'Max DD':>7s}  {'Min Eq':>9s}  {'Trades':>7s}  {'WR':>6s}",
          flush=True)
    print("-" * 75, flush=True)
    for rp in risk_levels:
        r = results_by_risk[rp]
        ret = r["pnl"] / START_EQUITY * 100
        print(f"{rp:>5.1%}  ${r['final_eq']:>8,.2f}  ${r['pnl']:>+10,.2f}  "
              f"{ret:>+7.1f}%  {r['max_dd_pct']:>6.1%}  "
              f"${r['min_equity']:>7,.2f}  {r['trades']:>7d}  "
              f"{r['wr']:>5.1f}%", flush=True)

    # Detailed output for each risk level
    for rp in risk_levels:
        r = results_by_risk[rp]
        ret = r["pnl"] / START_EQUITY * 100
        years = 3.0

        print("\n\n" + "=" * 75, flush=True)
        print(f"DETAILED: {rp:.1%} RISK PER TRADE", flush=True)
        print("=" * 75, flush=True)
        print(f"  Starting equity:      ${START_EQUITY:>10,.2f}", flush=True)
        print(f"  Final equity:         ${r['final_eq']:>10,.2f}", flush=True)
        print(f"  Total PnL:            ${r['pnl']:>+10,.2f} ({ret:+.1f}%)",
              flush=True)
        print(f"  Total fees:           ${r['total_fees']:>10,.2f}", flush=True)
        print(f"  Total funding (est):  ${r.get('total_funding', 0):>10,.2f}",
              flush=True)
        print(flush=True)
        print(f"  Trades executed:      {r['trades']:>8d}", flush=True)
        print(f"  Trades skipped:       {r['skipped']:>8d}", flush=True)
        print(f"  Win rate:             {r['wr']:>7.1f}%", flush=True)
        print(flush=True)
        print(f"  Max drawdown:         {r['max_dd_pct']:>7.1%}  "
              f"(${r['max_dd_usd']:>,.2f})", flush=True)
        print(f"  Lowest equity:        ${r['min_equity']:>10,.2f}  "
              f"({r['min_equity']/START_EQUITY*100:.1f}% of start)", flush=True)
        print(f"  Max concurrent pos:   {r['max_concurrent']:>8d}", flush=True)

        if r['final_eq'] > START_EQUITY:
            cagr = (r['final_eq'] / START_EQUITY) ** (1 / years) - 1
            print(f"  CAGR:                 {cagr:+.1%}", flush=True)

        # Equity milestones
        curve = r["equity_curve"]
        print(f"\n  Equity milestones:", flush=True)
        thresholds_hit = set()
        for ts, eq in curve:
            for threshold in [2000, 1500, 1000, 500, 250, 100]:
                if eq <= threshold and threshold not in thresholds_hit:
                    thresholds_hit.add(threshold)
                    print(f"    Dropped below ${threshold:,} at {str(ts)[:10]}  "
                          f"(equity: ${eq:,.2f})", flush=True)
            for threshold in [5000, 10000, 20000, 50000, 100000]:
                if eq >= threshold and threshold not in thresholds_hit:
                    thresholds_hit.add(threshold)
                    print(f"    Reached ${threshold:,} at {str(ts)[:10]}  "
                          f"(equity: ${eq:,.2f})", flush=True)

        # Risk assessment
        print(f"\n  RISK ASSESSMENT:", flush=True)
        if r["min_equity"] < START_EQUITY * 0.10:
            print(f"    !! NEARLY WIPED (dropped below 10%)", flush=True)
        elif r["min_equity"] < START_EQUITY * 0.25:
            print(f"    ! DANGER (dropped below 25%)", flush=True)
        elif r["min_equity"] < START_EQUITY * 0.50:
            print(f"    WARNING (dropped below 50%)", flush=True)
        else:
            print(f"    SAFE (never dropped below 50%)", flush=True)

    print("\n" + "=" * 75, flush=True)


if __name__ == "__main__":
    main()
