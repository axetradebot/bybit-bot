"""
Ablation: does the partial-TP ladder actually help the Sniper, or do
the extra fees from 4-part exits eat the partial-profit cushion?

Compares (everything else equal):
    A) Sniper WITH default ladder (30% @ 1.5R, 30% @ 3R, 20% @ 5R, 20% trail)
    B) Sniper with NO ladder (single TP + trailing stop, the v1 baseline)

Same symbols, same timeframes, same chop filter, same risk levels, same
candles.  Differences therefore isolate the ladder.

Usage:
    python scripts/partial_tp_ablation.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd

from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.backtest.live_aligned_portfolio import (
    collect_trades_live_aligned,
    run_portfolio_live_aligned,
)
from src.strategies.strategy_sniper import SniperStrategy
from src.strategies.base import _sf

START_EQUITY = 3_000.0
RISK_LEVELS = (0.01, 0.015, 0.02, 0.025, 0.03)
MAX_CONCURRENT = 4

SYMBOLS = ["SOLUSDT", "AVAXUSDT", "DOGEUSDT", "OPUSDT", "SUIUSDT"]
TIMEFRAMES = ["15m", "4h"]
FROM = "2023-01-01"
TO = "2026-04-03"


def _chop(bar, direction):
    """NEW-6 filter: DI + BB mid + vol>=1.2 + no squeeze (best in grid)."""
    p, m = _sf(bar.get("plus_di")), _sf(bar.get("minus_di"))
    if p > 0 or m > 0:
        if direction == "long" and p <= m:
            return False
        if direction == "short" and m <= p:
            return False
    bb_mid, close = _sf(bar.get("bb_mid")), _sf(bar.get("close"))
    if bb_mid > 0 and close > 0:
        if direction == "long" and close <= bb_mid:
            return False
        if direction == "short" and close >= bb_mid:
            return False
    vr = _sf(bar.get("volume_ratio"))
    if vr > 0 and vr < 1.2:
        return False
    if bar.get("bb_squeeze") is True:
        return False
    return True


def _make_sniper(*, ladder: bool, cooldown: int = 12) -> SniperStrategy:
    """Fresh strategy instance per (sym, tf) — Sniper's `_prev_close` /
    `_prev_ema21` are instance state and would otherwise leak across
    symbols, inflating signal counts dramatically."""
    s = SniperStrategy()
    s.cooldown_bars = cooldown
    if not ladder:
        s.default_tp_ladder = ()
        s.move_be_on_tp1 = False
    return s


def _collect(make_strategy, sym, tf, df_5m):
    ctx_df = build_bars_for_tf(df_5m, CONTEXT_TF.get(tf, "4h"))
    bars = build_bars_for_tf(df_5m, tf)
    rows = collect_trades_live_aligned(
        bars, ctx_df, sym, tf,
        snap_fn=lambda _b, _c: {},
        strategy=make_strategy(),
        chop_filter=_chop,
    )
    return [t for t, _ in rows]


def _summarise(label, results, start_eq):
    print(f"\n{label}")
    print("-" * 72)
    print(f"{'Risk':>6}  {'Final Eq':>10}  {'PnL':>12}  {'Return':>8}  "
          f"{'MaxDD':>7}  {'Trades':>6}  {'WR':>6}  {'Fees':>9}")
    print("-" * 72)
    for rp, r in results:
        ret = r["pnl"] / start_eq * 100
        print(f"{rp:>5.1%}  ${r['final_eq']:>8,.2f}  ${r['pnl']:>+10,.2f}  "
              f"{ret:>+7.1f}%  {r['max_dd_pct']:>6.1%}  {r['trades']:>6}  "
              f"{r['wr']:>5.1f}%  ${r['total_fees']:>7,.0f}")


def main():
    cache_dir = project_root / "data_cache"
    print("=" * 72)
    print("PARTIAL-TP LADDER ABLATION (Sniper, 5 symbols, 3yr)")
    print("=" * 72)

    print("\n[A] Sniper WITH ladder ((1.5,0.3),(3.0,0.3),(5.0,0.2)) cd=12")
    trades_a: list = []
    t0 = time.time()
    for sym in SYMBOLS:
        cache = cache_dir / f"{sym}_{FROM}_{TO}_5m.parquet"
        if not cache.exists():
            continue
        df_5m = pd.read_parquet(cache)
        for tf in TIMEFRAMES:
            chunk = _collect(lambda: _make_sniper(ladder=True, cooldown=12),
                             sym, tf, df_5m)
            trades_a.extend(chunk)
            print(f"  {tf:>3} {sym:<14} {len(chunk):>4} trades", flush=True)
    print(f"  -> {len(trades_a)} A-trades  ({time.time()-t0:.1f}s)")

    print("\n[B] Sniper WITHOUT ladder (single TP + trail) cd=12")
    trades_b: list = []
    t0 = time.time()
    for sym in SYMBOLS:
        cache = cache_dir / f"{sym}_{FROM}_{TO}_5m.parquet"
        if not cache.exists():
            continue
        df_5m = pd.read_parquet(cache)
        for tf in TIMEFRAMES:
            chunk = _collect(lambda: _make_sniper(ladder=False, cooldown=12),
                             sym, tf, df_5m)
            trades_b.extend(chunk)
            print(f"  {tf:>3} {sym:<14} {len(chunk):>4} trades", flush=True)
    print(f"  -> {len(trades_b)} B-trades  ({time.time()-t0:.1f}s)")

    print("\n[C] Sniper WITHOUT ladder cd=6 (matches historical best)")
    trades_c: list = []
    t0 = time.time()
    for sym in SYMBOLS:
        cache = cache_dir / f"{sym}_{FROM}_{TO}_5m.parquet"
        if not cache.exists():
            continue
        df_5m = pd.read_parquet(cache)
        for tf in TIMEFRAMES:
            chunk = _collect(lambda: _make_sniper(ladder=False, cooldown=6),
                             sym, tf, df_5m)
            trades_c.extend(chunk)
            print(f"  {tf:>3} {sym:<14} {len(chunk):>4} trades", flush=True)
    print(f"  -> {len(trades_c)} C-trades  ({time.time()-t0:.1f}s)")

    # --- Run portfolio at each risk level for each variant ---
    res_a, res_b, res_c = [], [], []
    for rp in RISK_LEVELS:
        ra = run_portfolio_live_aligned(
            trades_a, start_equity=START_EQUITY, risk_pct=rp,
            max_concurrent=MAX_CONCURRENT,
        )
        rb = run_portfolio_live_aligned(
            trades_b, start_equity=START_EQUITY, risk_pct=rp,
            max_concurrent=MAX_CONCURRENT,
        )
        rc = run_portfolio_live_aligned(
            trades_c, start_equity=START_EQUITY, risk_pct=rp,
            max_concurrent=MAX_CONCURRENT,
        )
        res_a.append((rp, ra))
        res_b.append((rp, rb))
        res_c.append((rp, rc))

    _summarise("VARIANT A -- WITH ladder (cd=12)", res_a, START_EQUITY)
    _summarise("VARIANT B -- NO ladder (cd=12)", res_b, START_EQUITY)
    _summarise("VARIANT C -- NO ladder (cd=6, historical-match)",
               res_c, START_EQUITY)

    # --- Per-risk delta vs winner ---
    print("\nDELTA  (best - A)  positive = ladder hurts")
    print("-" * 72)
    print(f"{'Risk':>6}  {'PnL d':>12}  {'MaxDD d':>9}  {'WR d':>7}  {'Fees d':>9}")
    print("-" * 72)
    for (rp, ra), (_, rb), (_, rc) in zip(res_a, res_b, res_c):
        # pick the better of B/C as baseline
        best = rb if rb["pnl"] >= rc["pnl"] else rc
        d_pnl = best["pnl"] - ra["pnl"]
        d_dd = best["max_dd_pct"] - ra["max_dd_pct"]
        d_wr = best["wr"] - ra["wr"]
        d_fees = best["total_fees"] - ra["total_fees"]
        print(f"{rp:>5.1%}  ${d_pnl:>+10,.2f}  {d_dd:>+8.1%}  "
              f"{d_wr:>+6.1f}%  ${d_fees:>+7,.0f}")


if __name__ == "__main__":
    main()
