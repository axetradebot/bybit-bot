"""
DEFINITIVE ANTI-CHOP FILTER SIMULATION

Runs the TOP recommended filter setups through the live-aligned engine
(identical fee/slippage/funding model to the real bot) and reports:
  - Full portfolio stats per filter at multiple risk levels
  - Per-symbol trade breakdown for the best config
  - Monthly PnL for the best config
  - Exit-reason distribution
  - Realism audit summary

Run:
    python scripts/best_filters_sim.py
    python scripts/best_filters_sim.py --start-equity 6000
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd

from src.backtest.live_aligned_portfolio import (
    LivePortfolioTrade,
    collect_trades_live_aligned,
    run_portfolio_live_aligned,
    MAKER_FEE,
    TAKER_FEE,
    SLIPPAGE_BPS,
    TRAIL_ACTIVATE,
    TRAIL_OFFSET,
    LEVERAGE,
    NOTIONAL_CAP_FRAC,
    LIMIT_ORDER_TTL,
    FUNDING_RATE_8H_FLAT,
    ENTRY_AGGRESSION_BPS,
    COOLDOWN_BARS,
)
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.base import _sf

DEFAULT_START_EQUITY = 3_000.0
RISK_GRID = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03)
MAX_CONCURRENT = 4

SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT",
    "1000PEPEUSDT", "DOGEUSDT", "OPUSDT", "SUIUSDT",
]
TIMEFRAMES = ["15m", "4h"]
FROM = "2023-01-01"
TO = "2026-04-03"


def _snap(bar: pd.Series, ctx: pd.Series) -> dict:
    return {
        "adx_14": _sf(bar.get("adx_14")),
        "plus_di": _sf(bar.get("plus_di")),
        "minus_di": _sf(bar.get("minus_di")),
        "bb_width": _sf(bar.get("bb_width")),
        "bb_squeeze": bool(bar.get("bb_squeeze", False)),
        "bb_mid": _sf(bar.get("bb_mid")),
        "volume_ratio": _sf(bar.get("volume_ratio")),
        "vwap": _sf(bar.get("vwap")),
        "close": _sf(bar.get("close")),
        "high": _sf(bar.get("high")),
        "low": _sf(bar.get("low")),
        "obv_slope": _sf(bar.get("obv_slope")),
        "candle_body_ratio": _sf(bar.get("candle_body_ratio")),
        "macd_hist": _sf(bar.get("macd_hist")),
        "macd_fast_hist": _sf(bar.get("macd_fast_hist")),
        "macd_hist_ctx": _sf(ctx.get("macd_hist")),
        "cmf_20": _sf(bar.get("cmf_20")),
        "roc_10": _sf(bar.get("roc_10")),
        "willr_14": _sf(bar.get("willr_14")),
        "mfi_14": _sf(bar.get("mfi_14")),
        "stochrsi_k": _sf(bar.get("stochrsi_k")),
        "atr_pct_rank": _sf(bar.get("atr_pct_rank")),
        "atr_14": _sf(bar.get("atr_14")),
    }


def load_candles(symbol, cache_dir):
    p = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Filter predicates
# ---------------------------------------------------------------------------

def di_aligned(s: dict, d: str) -> bool:
    p, m = s["plus_di"], s["minus_di"]
    if p <= 0 and m <= 0:
        return True
    return p > m if d == "long" else m > p


def bb_mid_side(s: dict, d: str) -> bool:
    return s["bb_mid"] > 0 and (
        (d == "long" and s["close"] > s["bb_mid"]) or
        (d == "short" and s["close"] < s["bb_mid"]))


def mfi_bias(s: dict, d: str) -> bool:
    return (d == "long" and s["mfi_14"] > 42) or (d == "short" and s["mfi_14"] < 58)


def bar_range_atr(s: dict, d: str) -> bool:
    return s["atr_14"] > 0 and (s["high"] - s["low"]) >= 0.45 * s["atr_14"]


def vwap_side(s: dict, d: str) -> bool:
    return s["vwap"] > 0 and (
        (d == "long" and s["close"] > s["vwap"]) or
        (d == "short" and s["close"] < s["vwap"]))


def vol_floor(s: dict, d: str, x: float) -> bool:
    v = s["volume_ratio"]
    return v >= x if v > 0 else True


def no_squeeze(s: dict, d: str) -> bool:
    return not s["bb_squeeze"]


def macd_aligned(hist: float, d: str) -> bool:
    return hist > 0 if d == "long" else hist < 0


def atr_rank_floor(s: dict, d: str, x: float) -> bool:
    return s["atr_pct_rank"] >= x


def adx_floor(s: dict, d: str, x: float) -> bool:
    a = s["adx_14"]
    return a >= x if a > 0 else True


# ---------------------------------------------------------------------------
# Recommended scenarios (based on prior sweep results)
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("BASELINE (no filter)",
     lambda s, d: True),

    # --- TOP WINNERS from prior sims ---
    ("REC-1: DI + BB mid + ATR rank>=0.35",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d) and atr_rank_floor(s, d, 0.35)),

    ("REC-2: DI + BB mid",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d)),

    ("REC-3: volume_ratio >= 1.2",
     lambda s, d: vol_floor(s, d, 1.2)),

    ("REC-4: DI + vol>=1.2",
     lambda s, d: di_aligned(s, d) and vol_floor(s, d, 1.2)),

    ("REC-5: DI + MFI bias",
     lambda s, d: di_aligned(s, d) and mfi_bias(s, d)),

    ("REC-6: DI + BB mid + MFI",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d) and mfi_bias(s, d)),

    ("REC-7: DI + BB mid + bar range",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d) and bar_range_atr(s, d)),

    ("REC-8: DI + BB mid + MFI + vol>=1.0",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d) and mfi_bias(s, d) and vol_floor(s, d, 1.0)),

    ("REC-9: DI + BB mid + MFI + bar range",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d) and mfi_bias(s, d) and bar_range_atr(s, d)),

    ("REC-10: BB mid + MFI + bar range",
     lambda s, d: bb_mid_side(s, d) and mfi_bias(s, d) and bar_range_atr(s, d)),

    # --- EXTRA: new untested combos for deeper chop filtering ---
    ("NEW-1: DI + BB mid + ATR rank>=0.35 + MACD 5m",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d)
     and atr_rank_floor(s, d, 0.35) and macd_aligned(s["macd_hist"], d)),

    ("NEW-2: DI + BB mid + ATR rank>=0.35 + vol>=1.0",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d)
     and atr_rank_floor(s, d, 0.35) and vol_floor(s, d, 1.0)),

    ("NEW-3: DI + BB mid + ATR rank>=0.35 + no squeeze",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d)
     and atr_rank_floor(s, d, 0.35) and no_squeeze(s, d)),

    ("NEW-4: DI + BB mid + ATR rank>=0.35 + VWAP",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d)
     and atr_rank_floor(s, d, 0.35) and vwap_side(s, d)),

    ("NEW-5: DI + BB mid + MFI + ATR rank>=0.35",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d)
     and mfi_bias(s, d) and atr_rank_floor(s, d, 0.35)),

    ("NEW-6: DI + BB mid + vol>=1.2 + no squeeze",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d)
     and vol_floor(s, d, 1.2) and no_squeeze(s, d)),

    ("NEW-7: ADX>=20 + DI + BB mid + ATR rank>=0.35",
     lambda s, d: adx_floor(s, d, 20) and di_aligned(s, d)
     and bb_mid_side(s, d) and atr_rank_floor(s, d, 0.35)),

    ("NEW-8: DI + BB mid + bar range + MACD 5m",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d)
     and bar_range_atr(s, d) and macd_aligned(s["macd_hist"], d)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _monthly_pnl(trades: list[LivePortfolioTrade], risk_pct: float,
                 start_equity: float = DEFAULT_START_EQUITY) -> list[dict]:
    """Run the portfolio and bucket PnL by calendar month."""
    r = run_portfolio_live_aligned(
        trades, start_equity=start_equity, risk_pct=risk_pct,
        max_concurrent=MAX_CONCURRENT, track_equity_curve=True,
        curve_from=FROM,
    )
    curve = r.get("equity_curve", [])
    if len(curve) < 2:
        return []
    months: dict[str, dict] = {}
    prev_eq = start_equity
    for ts, eq in curve:
        key = pd.Timestamp(ts).strftime("%Y-%m")
        if key not in months:
            months[key] = {"start": prev_eq, "end": eq, "trades": 0}
        months[key]["end"] = eq
        months[key]["trades"] += 1
        prev_eq = eq
    out = []
    for m in sorted(months):
        d = months[m]
        pnl = d["end"] - d["start"]
        pct = pnl / d["start"] * 100 if d["start"] > 0 else 0
        out.append({"month": m, "pnl": pnl, "pct": pct,
                     "end_eq": d["end"], "trades": d["trades"]})
    return out


def _per_symbol_stats(
    rows: list[tuple[LivePortfolioTrade, dict]],
    pred,
) -> dict[str, dict]:
    by_sym: dict[str, list] = defaultdict(list)
    for t, snap in rows:
        if pred(snap, t.direction):
            by_sym[t.symbol].append(t)
    out = {}
    for sym in sorted(by_sym):
        ts = by_sym[sym]
        wins = sum(1 for t in ts if t.raw_pct > 0)
        n = len(ts)
        avg_pct = sum(t.raw_pct for t in ts) / n if n else 0
        out[sym] = {"n": n, "wins": wins, "wr": wins / n * 100 if n else 0,
                     "avg_raw_pct": avg_pct * 100}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-equity", type=float, default=DEFAULT_START_EQUITY,
                        help="Starting equity in USD (default %(default)s)")
    args = parser.parse_args()
    START_EQUITY = args.start_equity

    cache_dir = project_root / "data_cache"
    w = 120

    print("=" * w, flush=True)
    print("DEFINITIVE ANTI-CHOP FILTER SIMULATION", flush=True)
    print("=" * w, flush=True)

    # ---- Realism audit ----
    print("\n  REALISM AUDIT (sim vs live bot)", flush=True)
    print(f"    Entry:            Aggressive limit (close +/-{ENTRY_AGGRESSION_BPS:.2%}), "
          f"TTL = {LIMIT_ORDER_TTL} bars, adverse slippage {SLIPPAGE_BPS:.2%} on fill", flush=True)
    print(f"    Entry fee:        TAKER {TAKER_FEE:.3%} (aggressive limit pays taker on Bybit)", flush=True)
    print(f"    Exit fee (TP):    MAKER {MAKER_FEE:.2%} (limit TP)", flush=True)
    print(f"    Exit fee (SL/trail/timeout): TAKER {TAKER_FEE:.3%} + slippage {SLIPPAGE_BPS:.2%}", flush=True)
    print(f"    Funding:          Pro-rata by 8h periods held; flat rate {FUNDING_RATE_8H_FLAT} "
          "when per-bar data missing", flush=True)
    print(f"    Sizing:           risk% of equity / (SL_dist + entry * round_trip_cost)", flush=True)
    print(f"    Notional cap:     {NOTIONAL_CAP_FRAC:.0%} of equity * {LEVERAGE}x leverage", flush=True)
    print(f"    Trail:            activate at +{TRAIL_ACTIVATE:.0%}, offset {TRAIL_OFFSET:.0%}", flush=True)
    print(f"    Cooldown:         {COOLDOWN_BARS} bars between exits and new entries per symbol", flush=True)
    print(f"    Max concurrent:   {MAX_CONCURRENT} positions", flush=True)
    print(f"    Start equity:     ${START_EQUITY:,.0f}", flush=True)
    print(f"    Symbols:          {', '.join(SYMBOLS)}", flush=True)
    print(f"    Timeframes:       {', '.join(TIMEFRAMES)}", flush=True)
    print(f"    Period:           {FROM} -> {TO}", flush=True)
    print(f"    Risk grid:        {RISK_GRID}", flush=True)

    # ---- Collect trades ----
    print(f"\n{'='*w}", flush=True)
    print("PHASE 1: Collecting baseline trades + indicator snapshots", flush=True)
    print("=" * w, flush=True)
    rows: list[tuple[LivePortfolioTrade, dict]] = []
    sym_counts: dict[str, int] = {}
    for sym in SYMBOLS:
        df_5m = load_candles(sym, cache_dir)
        if df_5m.empty:
            print(f"  SKIP {sym} (no cached data)", flush=True)
            continue
        sym_total = 0
        for tf in TIMEFRAMES:
            t0 = time.time()
            ctx_df = build_bars_for_tf(df_5m, CONTEXT_TF.get(tf, "4h"))
            bars = build_bars_for_tf(df_5m, tf)
            chunk = collect_trades_live_aligned(
                bars, ctx_df, sym, tf, snap_fn=lambda b, c: _snap(b, c),
            )
            rows.extend(chunk)
            sym_total += len(chunk)
            print(f"  {tf:>4s} {sym:<16s}: {len(chunk):>5d} trades  ({time.time()-t0:.1f}s)",
                  flush=True)
        sym_counts[sym] = sym_total

    rows.sort(key=lambda x: x[0].entry_time)
    n_base = len(rows)
    print(f"\n  TOTAL baseline trades: {n_base:,}", flush=True)

    # ---- Grid sweep ----
    print(f"\n{'='*w}", flush=True)
    print(f"PHASE 2: Grid search — {len(SCENARIOS)} scenarios x {len(RISK_GRID)} risk levels "
          f"= {len(SCENARIOS)*len(RISK_GRID)} configs", flush=True)
    print("=" * w, flush=True)

    results: list[dict] = []
    t_grid = time.time()
    for label, pred in SCENARIOS:
        filt = [t for t, snap in rows if pred(snap, t.direction)]
        for rp in RISK_GRID:
            r = run_portfolio_live_aligned(
                filt, start_equity=START_EQUITY, risk_pct=rp,
                max_concurrent=MAX_CONCURRENT,
            )
            results.append({
                "label": label, "risk_pct": rp, "kept": len(filt),
                "final_eq": r["final_eq"], "pnl": r["pnl"],
                "dd": r["max_dd_pct"], "dd_usd": r["max_dd_usd"],
                "trades": r["trades"], "wr": r["wr"],
                "wins": r["wins"],
                "fees": r["total_fees"], "funding": r["total_funding"],
                "tp": r["tp"], "sl": r["sl"],
                "trail": r["trail"], "timeout": r["timeout"],
                "min_eq": r["min_equity"], "skipped": r["skipped"],
            })
    elapsed = time.time() - t_grid
    print(f"  Evaluated {len(results)} configs in {elapsed:.1f}s", flush=True)

    # ---- Rank by final equity ----
    results.sort(key=lambda x: x["final_eq"], reverse=True)

    print(f"\n{'='*w}", flush=True)
    print("PHASE 3: RESULTS — ranked by FINAL EQUITY (max PnL)", flush=True)
    print("=" * w, flush=True)

    best = results[0]
    print(f"\n  >>> BEST CONFIG <<<", flush=True)
    print(f"  Filter:      {best['label']}", flush=True)
    print(f"  Risk/trade:  {best['risk_pct']:.2%}", flush=True)
    print(f"  Final eq:    ${best['final_eq']:,.0f}", flush=True)
    print(f"  Total PnL:   ${best['pnl']:+,.0f}  ({best['pnl']/START_EQUITY*100:+.0f}%)", flush=True)
    print(f"  Max DD:      {best['dd']:.1%}  (${best['dd_usd']:,.0f})", flush=True)
    print(f"  Trades:      {best['trades']:,}  (W {best['wins']:,} / L {best['trades']-best['wins']:,})",
          flush=True)
    print(f"  Win rate:    {best['wr']:.1f}%", flush=True)
    print(f"  Fees paid:   ${best['fees']:,.0f}", flush=True)
    print(f"  Funding:     ${best['funding']:,.0f}", flush=True)
    print(f"  Exits:       TP {best['tp']}  SL {best['sl']}  Trail {best['trail']}  "
          f"Timeout {best['timeout']}", flush=True)
    print(f"  Skipped:     {best['skipped']}", flush=True)

    # ---- Top 30 table ----
    print(f"\n{'='*w}", flush=True)
    print("TOP 30 CONFIGS BY FINAL EQUITY", flush=True)
    hdr = (f"{'#':>3}  {'Risk':>6}  {'Final$':>10}  {'PnL':>11}  {'Ret%':>7}  "
           f"{'MaxDD':>7}  {'WR':>6}  {'Tr':>5}  {'Fees':>8}  {'Fund':>7}  {'Filter':<45}")
    print(hdr, flush=True)
    print("-" * w, flush=True)
    for i, x in enumerate(results[:30], 1):
        ret = x["pnl"] / START_EQUITY * 100
        print(
            f"{i:>3}  {x['risk_pct']:>5.2%}  ${x['final_eq']:>8,.0f}  "
            f"${x['pnl']:>+9,.0f}  {ret:>+6.0f}%  {x['dd']:>6.1%}  "
            f"{x['wr']:>5.1f}%  {x['trades']:>5}  ${x['fees']:>6,.0f}  "
            f"${x['funding']:>5,.0f}  {x['label'][:45]}",
            flush=True,
        )

    # ---- Best per-filter (pick best risk for each filter) ----
    print(f"\n{'='*w}", flush=True)
    print("BEST RISK LEVEL PER FILTER (one row per filter, best risk chosen)", flush=True)
    hdr2 = (f"{'#':>3}  {'Risk':>6}  {'Final$':>10}  {'PnL':>11}  "
            f"{'MaxDD':>7}  {'WR':>6}  {'Tr':>5}  {'Filter':<50}")
    print(hdr2, flush=True)
    print("-" * w, flush=True)
    seen_labels: set[str] = set()
    rank = 0
    for x in results:
        if x["label"] in seen_labels:
            continue
        seen_labels.add(x["label"])
        rank += 1
        print(
            f"{rank:>3}  {x['risk_pct']:>5.2%}  ${x['final_eq']:>8,.0f}  "
            f"${x['pnl']:>+9,.0f}  {x['dd']:>6.1%}  "
            f"{x['wr']:>5.1f}%  {x['trades']:>5}  {x['label'][:50]}",
            flush=True,
        )

    # ---- Per-symbol breakdown for #1 ----
    best_label = best["label"]
    best_pred = None
    for label, pred in SCENARIOS:
        if label == best_label:
            best_pred = pred
            break

    if best_pred:
        print(f"\n{'='*w}", flush=True)
        print(f"PER-SYMBOL BREAKDOWN for: {best_label}", flush=True)
        print(f"{'Symbol':<18}  {'Signals':>8}  {'Wins':>6}  {'WR':>6}  {'AvgRaw%':>8}", flush=True)
        print("-" * 55, flush=True)
        sym_stats = _per_symbol_stats(rows, best_pred)
        for sym in SYMBOLS:
            if sym in sym_stats:
                ss = sym_stats[sym]
                print(f"{sym:<18}  {ss['n']:>8}  {ss['wins']:>6}  "
                      f"{ss['wr']:>5.1f}%  {ss['avg_raw_pct']:>+7.2f}%", flush=True)

    # ---- Monthly PnL for #1 ----
    if best_pred:
        best_risk = best["risk_pct"]
        filt_trades = [t for t, snap in rows if best_pred(snap, t.direction)]
        monthly = _monthly_pnl(filt_trades, best_risk, START_EQUITY)
        if monthly:
            print(f"\n{'='*w}", flush=True)
            print(f"MONTHLY PnL for: {best_label}  @ risk {best_risk:.2%}", flush=True)
            print(f"{'Month':<10}  {'PnL':>10}  {'%':>7}  {'EndEq':>10}  {'Trades':>7}", flush=True)
            print("-" * 50, flush=True)
            for m in monthly:
                print(f"{m['month']:<10}  ${m['pnl']:>+8,.0f}  {m['pct']:>+6.1f}%  "
                      f"${m['end_eq']:>8,.0f}  {m['trades']:>7}", flush=True)

    # ---- Exit reason distribution for top 5 filters ----
    print(f"\n{'='*w}", flush=True)
    print("EXIT REASON DISTRIBUTION (top 5 filters at their best risk)", flush=True)
    print(f"{'Filter':<45}  {'TP':>5}  {'SL':>5}  {'Trail':>5}  {'T/O':>5}  {'TP%':>6}", flush=True)
    print("-" * 80, flush=True)
    seen2: set[str] = set()
    cnt = 0
    for x in results:
        if x["label"] in seen2:
            continue
        seen2.add(x["label"])
        cnt += 1
        if cnt > 5:
            break
        total_exits = x["tp"] + x["sl"] + x["trail"] + x["timeout"]
        tp_pct = x["tp"] / total_exits * 100 if total_exits else 0
        print(f"{x['label'][:45]:<45}  {x['tp']:>5}  {x['sl']:>5}  "
              f"{x['trail']:>5}  {x['timeout']:>5}  {tp_pct:>5.1f}%", flush=True)

    print(f"\n{'='*w}", flush=True)
    print("SIMULATION COMPLETE", flush=True)
    print("=" * w, flush=True)


if __name__ == "__main__":
    main()
