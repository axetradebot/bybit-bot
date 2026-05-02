"""
Per-symbol profitability scan under the CURRENT LIVE config.

For each candidate symbol, run signal collection + trade resolution at
   sl=2.5x ATR, tp=13.5x ATR, R:R=5.4
   risk_pct=0.25%, equity=$3,569 (mirrors prod), leverage=20x
on a clean 3-year window (2023-04-01 -> 2026-04-01).

Each symbol gets its OWN equity sim (no shared portfolio) so the
include/exclude verdict is purely about that symbol's standalone edge.

Output: ranked table with trade count, WR, total PnL, avg PnL/trade,
        expectancy in R, max drawdown, and a sub-window (last 12 months)
        sanity check so we don't include something that worked great in
        2023 but rolled over in 2025-2026.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from scripts.sl_width_sim_fast import (
    PrecomputedTrade,
    collect_signals,
    resolve_trades,
    MAKER_FEE,
    TAKER_FEE,
    ROUND_TRIP_COST,
    LEVERAGE,
    NOTIONAL_CAP_FRAC,
)

# ── Config matches LIVE ──────────────────────────────────────
SL_MULT = 2.5
TP_MULT = 13.5
RISK_PCT = 0.0025
START_EQUITY = 3569.0  # current live equity
MAX_HOLD_BARS_15M = 96   # 24h on 15m
MAX_HOLD_BARS_4H = 6     # 24h on 4h
TIMEFRAMES = ["15m", "4h"]

DATE_RANGE = ("2023-04-01", "2026-04-01")
SUBWINDOW_DAYS = 365  # "last 12 months" sanity slice

# ── Candidate symbols (have 2023-04-01 -> 2026-04-01 cache) ──
LIVE_NOW = {
    "SOLUSDT", "AVAXUSDT", "WIFUSDT",
    "DOGEUSDT", "BTCUSDT", "XRPUSDT",
}
ALL_CANDIDATES = [
    # Live 6
    "BTCUSDT", "SOLUSDT", "AVAXUSDT", "WIFUSDT", "DOGEUSDT", "XRPUSDT",
    # User-flagged readd candidates
    "OPUSDT", "1000PEPEUSDT",
    # Other majors with data
    "ETHUSDT", "LINKUSDT", "NEARUSDT", "AAVEUSDT", "ARBUSDT", "APTUSDT",
    "LTCUSDT", "DOTUSDT", "INJUSDT", "SUIUSDT", "TIAUSDT",
    "RUNEUSDT", "SEIUSDT", "STXUSDT", "FETUSDT", "RENDERUSDT",
    "ENAUSDT", "JUPUSDT", "ONDOUSDT", "WUSDT",
]


def load_5m(symbol: str) -> pd.DataFrame:
    """Pick the best available cache (3yr first, then any other)."""
    cache_dir = project_root / "data_cache"
    candidates = [
        cache_dir / f"{symbol}_{DATE_RANGE[0]}_{DATE_RANGE[1]}_5m.parquet",
        cache_dir / f"{symbol}_2023-01-01_2026-04-03_5m.parquet",
        cache_dir / f"{symbol}_2024-01-01_2026-04-29_5m.parquet",
    ]
    for c in candidates:
        if c.exists():
            return pd.read_parquet(c), c.name
    return pd.DataFrame(), None


def per_symbol_portfolio(trades: list[PrecomputedTrade],
                         start_eq: float = START_EQUITY) -> dict:
    """Single-equity sim: this symbol vs. start_eq, sequential trades."""
    if not trades:
        return {"trades": 0, "wins": 0, "wr": 0.0, "pnl": 0.0,
                "final_eq": start_eq, "max_dd_pct": 0.0,
                "expectancy_r": 0.0, "tp": 0, "sl": 0, "trail": 0,
                "timeout": 0, "avg_pnl": 0.0,
                "first_trade": None, "last_trade": None}

    equity = start_eq
    peak = equity
    max_dd = 0.0
    pnl_total = 0.0
    wins = 0
    rs: list[float] = []  # per-trade R (PnL / dollar-risk)
    counts = {"tp": 0, "sl": 0, "trail": 0, "timeout": 0}

    for t in sorted(trades, key=lambda x: x.entry_time):
        if equity <= 0:
            continue
        risk_amount = equity * RISK_PCT
        qty = risk_amount / (t.sl_distance + t.entry_price * ROUND_TRIP_COST)
        notional = qty * t.entry_price
        max_notional = equity * LEVERAGE * NOTIONAL_CAP_FRAC
        if notional > max_notional:
            notional = max_notional

        entry_fee = notional * MAKER_FEE
        exit_fee = notional * TAKER_FEE
        pnl = notional * t.raw_pct - entry_fee - exit_fee
        equity += pnl
        pnl_total += pnl
        if pnl > 0:
            wins += 1
        rs.append(pnl / risk_amount if risk_amount > 0 else 0)
        if t.exit_reason in counts:
            counts[t.exit_reason] += 1

        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    n = len(trades)
    return {
        "trades": n, "wins": wins, "wr": wins / n * 100,
        "pnl": pnl_total, "final_eq": equity,
        "max_dd_pct": max_dd, "expectancy_r": float(np.mean(rs)) if rs else 0,
        "avg_pnl": pnl_total / n,
        "tp": counts["tp"], "sl": counts["sl"],
        "trail": counts["trail"], "timeout": counts["timeout"],
        "first_trade": min(t.entry_time for t in trades),
        "last_trade": max(t.entry_time for t in trades),
    }


def evaluate_symbol(symbol: str) -> dict:
    df_5m, src = load_5m(symbol)
    if df_5m.empty:
        return {"symbol": symbol, "error": "no cache"}

    # All trades across both TFs, run single-equity sim per TF then sum.
    # (Matches live where TFs run parallel against shared equity, but
    # for an include/exclude decision per-TF separation is simpler and
    # the cross-TF interaction is small given the cooldown.)
    trades_all: list[PrecomputedTrade] = []
    sigs_per_tf = {}
    for tf in TIMEFRAMES:
        ctx_tf = CONTEXT_TF.get(tf, "4h")
        ctx_df = build_bars_for_tf(df_5m, ctx_tf)
        trading = build_bars_for_tf(df_5m, tf)
        sigs = collect_signals(trading, ctx_df, symbol, tf)
        sigs_per_tf[tf] = (len(sigs), trading)
        trades = resolve_trades(sigs, trading, SL_MULT, TP_MULT, symbol, tf)
        trades_all.extend(trades)

    full = per_symbol_portfolio(trades_all)
    full["symbol"] = symbol
    full["src"] = src
    full["signals_15m"] = sigs_per_tf.get("15m", (0, None))[0]
    full["signals_4h"] = sigs_per_tf.get("4h", (0, None))[0]

    # Last 12-month sub-window: filter trades, re-run portfolio.
    # Normalize tz to whatever entry_time uses (PrecomputedTrade
    # carries tz-naive timestamps; cutoff must match).
    cutoff_utc = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=SUBWINDOW_DAYS)
    sample_ts = trades_all[0].entry_time if trades_all else None
    if sample_ts is not None and sample_ts.tzinfo is None:
        cutoff = cutoff_utc.tz_localize(None)
    else:
        cutoff = cutoff_utc
    recent_trades = [t for t in trades_all if t.entry_time >= cutoff]
    sub = per_symbol_portfolio(recent_trades)
    full["sub_trades"] = sub["trades"]
    full["sub_wr"] = sub["wr"]
    full["sub_pnl"] = sub["pnl"]
    full["sub_dd"] = sub["max_dd_pct"]
    return full


def fmt_row(r: dict) -> str:
    if "error" in r:
        return f"  {r['symbol']:<14s}  [no data]"
    live_marker = " *" if r["symbol"] in LIVE_NOW else "  "
    return (
        f"{live_marker}{r['symbol']:<14s} "
        f"{r['trades']:>5d}  {r['wr']:>5.1f}%  "
        f"${r['pnl']:>+8,.0f}  ${r['avg_pnl']:>+6.2f}  "
        f"{r['expectancy_r']:>+5.2f}R  "
        f"{r['max_dd_pct']:>5.1%}  "
        f"{r['tp']:>3d}/{r['sl']:>3d}/{r['trail']:>3d}/{r['timeout']:>3d}  "
        f"|  {r['sub_trades']:>4d}  {r['sub_wr']:>5.1f}%  "
        f"${r['sub_pnl']:>+8,.0f}  {r['sub_dd']:>5.1%}"
    )


def main():
    print("=" * 130, flush=True)
    print(f"PER-SYMBOL EVALUATION  |  config: SL={SL_MULT}x ATR  "
          f"TP={TP_MULT}x ATR  R:R={TP_MULT/SL_MULT:.2f}  "
          f"risk={RISK_PCT*100:.2f}%  equity=${START_EQUITY:,.0f}",
          flush=True)
    print(f"Date range: {DATE_RANGE[0]} -> {DATE_RANGE[1]}  ({SUBWINDOW_DAYS}d sub-window)",
          flush=True)
    print("=" * 130, flush=True)

    results = []
    for i, sym in enumerate(ALL_CANDIDATES, 1):
        t0 = time.time()
        print(f"  [{i:>2d}/{len(ALL_CANDIDATES)}] {sym} ...",
              end=" ", flush=True)
        r = evaluate_symbol(sym)
        results.append(r)
        if "error" in r:
            print(f"SKIP ({r['error']})", flush=True)
        else:
            print(f"{r['trades']} trades, "
                  f"${r['pnl']:+,.0f} ({time.time()-t0:.0f}s)",
                  flush=True)

    valid = [r for r in results if "error" not in r]
    valid.sort(key=lambda r: r["pnl"], reverse=True)

    print("\n" + "=" * 130, flush=True)
    print(f"  {'SYM':<14s} {'Trades':>5s}  {'WR':>5s}  "
          f"{'PnL':>9s}  {'Avg/T':>8s}  {'Exp':>6s}  "
          f"{'MaxDD':>6s}  {'TP/SL/TR/TO':>15s}  "
          f"|  {'12m T':>4s}  {'12m WR':>6s}  "
          f"{'12m PnL':>9s}  {'12m DD':>6s}", flush=True)
    print("-" * 130, flush=True)
    for r in valid:
        print(fmt_row(r), flush=True)

    # Recommendation summary
    print("\n" + "=" * 130, flush=True)
    print("RECOMMENDATION", flush=True)
    print("=" * 130, flush=True)
    keep_live = [r for r in valid if r["symbol"] in LIVE_NOW
                 and r["pnl"] > 0 and r["sub_pnl"] > 0]
    drop_live = [r for r in valid if r["symbol"] in LIVE_NOW
                 and (r["pnl"] <= 0 or r["sub_pnl"] <= 0)]
    add_new = [r for r in valid if r["symbol"] not in LIVE_NOW
               and r["pnl"] > 0 and r["sub_pnl"] > 0
               and r["trades"] >= 50 and r["sub_trades"] >= 10]
    add_new.sort(key=lambda r: r["sub_pnl"], reverse=True)

    print(f"\n  KEEP (live, profitable on 3yr AND 12m):", flush=True)
    for r in keep_live:
        print(f"    {r['symbol']:<14s}  3yr ${r['pnl']:+,.0f}  "
              f"12m ${r['sub_pnl']:+,.0f}", flush=True)

    print(f"\n  CONSIDER DROPPING (live, unprofitable):", flush=True)
    for r in drop_live:
        print(f"    {r['symbol']:<14s}  3yr ${r['pnl']:+,.0f}  "
              f"12m ${r['sub_pnl']:+,.0f}", flush=True)

    print(f"\n  CONSIDER ADDING (not live, profitable + sufficient trades):",
          flush=True)
    for r in add_new:
        print(f"    {r['symbol']:<14s}  3yr ${r['pnl']:+,.0f} ({r['trades']}t "
              f"WR {r['wr']:.0f}%)  12m ${r['sub_pnl']:+,.0f} "
              f"({r['sub_trades']}t WR {r['sub_wr']:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
