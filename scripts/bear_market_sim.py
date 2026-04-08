"""
BEAR MARKET OPTIMIZATION SIMULATION

Runs the expanded 10-symbol universe across three time windows:
  1. FULL period   (2023-01-01 -> 2026-04-03)
  2. BEAR-2023     (2023-01-01 -> 2023-12-31) — sustained altcoin chop
  3. BEAR-LATE     (2025-11-01 -> 2026-04-03) — current crash regime

Sweeps over:
  - Filter scenarios (baseline + bear-tuned combos)
  - Risk per trade   (0.5% .. 3%)
  - Max concurrent   (2 or 4)
  - Volume floor     (1.0x, 1.2x, 1.5x, 2.0x)

Reports:
  - Best configs that SURVIVE bear windows (positive or min loss)
    while still delivering PnL on the full period
  - Per-symbol contribution so we can prune weak symbols
  - Monthly PnL for the best bear-surviving config

Run:
    python scripts/bear_market_sim.py
    python scripts/bear_market_sim.py --start-equity 6000
"""

from __future__ import annotations

import argparse
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

SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT",
    "1000PEPEUSDT", "DOGEUSDT", "OPUSDT", "SUIUSDT",
    "BTCUSDT", "ETHUSDT", "XRPUSDT",
]
TIMEFRAMES = ["15m", "4h"]
FROM = "2023-01-01"
TO = "2026-04-03"

WINDOWS = {
    "FULL":      (FROM, TO),
    "BEAR-2023": ("2023-01-01", "2023-12-31"),
    "BEAR-LATE": ("2025-11-01", "2026-04-03"),
}

RISK_GRID = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.025, 0.03)
MAX_CONCURRENT_GRID = (2, 4)
VOL_FLOOR_GRID = (1.0, 1.2, 1.5, 2.0)


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


def load_candles(symbol: str, cache_dir: Path) -> pd.DataFrame:
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
# Scenarios: the volume floor is applied separately as a sweep dimension
# ---------------------------------------------------------------------------

FILTER_SCENARIOS = [
    ("BASELINE",
     lambda s, d: True),

    ("LIVE: DI+BBmid+vol+noSqz",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d) and no_squeeze(s, d)),

    ("DI+BBmid",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d)),

    ("DI+BBmid+ATR>=0.35",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d) and atr_rank_floor(s, d, 0.35)),

    ("DI+BBmid+ATR>=0.40",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d) and atr_rank_floor(s, d, 0.40)),

    ("DI+BBmid+ATR>=0.50",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d) and atr_rank_floor(s, d, 0.50)),

    ("ADX>=20+DI+BBmid",
     lambda s, d: adx_floor(s, d, 20) and di_aligned(s, d) and bb_mid_side(s, d)),

    ("ADX>=25+DI+BBmid",
     lambda s, d: adx_floor(s, d, 25) and di_aligned(s, d) and bb_mid_side(s, d)),

    ("ADX>=20+DI+BBmid+ATR>=0.35",
     lambda s, d: adx_floor(s, d, 20) and di_aligned(s, d) and bb_mid_side(s, d)
     and atr_rank_floor(s, d, 0.35)),

    ("DI+BBmid+MFI",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d) and mfi_bias(s, d)),

    ("DI+BBmid+MACD",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d)
     and macd_aligned(s["macd_hist"], d)),

    ("DI+BBmid+noSqz+ATR>=0.35",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d) and no_squeeze(s, d)
     and atr_rank_floor(s, d, 0.35)),

    ("DI+BBmid+noSqz+MACD",
     lambda s, d: di_aligned(s, d) and bb_mid_side(s, d) and no_squeeze(s, d)
     and macd_aligned(s["macd_hist"], d)),

    ("ADX>=25+DI+BBmid+ATR>=0.40",
     lambda s, d: adx_floor(s, d, 25) and di_aligned(s, d) and bb_mid_side(s, d)
     and atr_rank_floor(s, d, 0.40)),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_trades_by_window(
    trades: list[tuple[LivePortfolioTrade, dict]],
    win_start: str,
    win_end: str,
) -> list[tuple[LivePortfolioTrade, dict]]:
    ws = pd.Timestamp(win_start, tz="UTC")
    we = pd.Timestamp(win_end, tz="UTC")
    out = []
    for t, s in trades:
        et = t.entry_time
        if et.tzinfo is None:
            et = et.tz_localize("UTC")
        if ws <= et <= we:
            out.append((t, s))
    return out


def _monthly_pnl(trades: list[LivePortfolioTrade], risk_pct: float,
                 max_concurrent: int, start_equity: float,
                 curve_from: str) -> list[dict]:
    r = run_portfolio_live_aligned(
        trades, start_equity=start_equity, risk_pct=risk_pct,
        max_concurrent=max_concurrent, track_equity_curve=True,
        curve_from=curve_from,
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
    vol_floor_val: float,
) -> dict[str, dict]:
    by_sym: dict[str, list] = defaultdict(list)
    for t, snap in rows:
        if pred(snap, t.direction) and vol_floor(snap, t.direction, vol_floor_val):
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
    w = 130

    print("=" * w, flush=True)
    print("BEAR MARKET OPTIMIZATION SIMULATION", flush=True)
    print("=" * w, flush=True)

    print("\n  REALISM AUDIT (sim vs live bot)", flush=True)
    print(f"    Entry fee:    TAKER {TAKER_FEE:.3%}  |  Exit TP: MAKER {MAKER_FEE:.2%}  |  "
          f"Exit SL/trail: TAKER {TAKER_FEE:.3%} + slip {SLIPPAGE_BPS:.2%}", flush=True)
    print(f"    Sizing:       risk% / (SL_dist + entry * {TAKER_FEE+TAKER_FEE+SLIPPAGE_BPS:.4%})", flush=True)
    print(f"    Notional cap: {NOTIONAL_CAP_FRAC:.0%} of equity * {LEVERAGE}x", flush=True)
    print(f"    Trail:        activate +{TRAIL_ACTIVATE:.0%}, offset {TRAIL_OFFSET:.0%}", flush=True)
    print(f"    Cooldown:     {COOLDOWN_BARS} bars  |  Limit TTL: {LIMIT_ORDER_TTL} bars", flush=True)
    print(f"    Funding:      flat {FUNDING_RATE_8H_FLAT} per 8h when per-bar data missing", flush=True)
    print(f"    Start equity: ${START_EQUITY:,.0f}", flush=True)
    print(f"    Symbols:      {', '.join(SYMBOLS)}", flush=True)
    print(f"    Period:       {FROM} -> {TO}", flush=True)
    print(f"    Windows:      {list(WINDOWS.keys())}", flush=True)
    print(f"    Filters:      {len(FILTER_SCENARIOS)}", flush=True)
    print(f"    Risk grid:    {RISK_GRID}", flush=True)
    print(f"    MaxConc grid: {MAX_CONCURRENT_GRID}", flush=True)
    print(f"    VolFloor grid:{VOL_FLOOR_GRID}", flush=True)

    total_configs = (len(FILTER_SCENARIOS) * len(RISK_GRID)
                     * len(MAX_CONCURRENT_GRID) * len(VOL_FLOOR_GRID)
                     * len(WINDOWS))
    print(f"    Total evals:  {total_configs:,}", flush=True)

    # ---- Collect trades ----
    print(f"\n{'='*w}", flush=True)
    print("PHASE 1: Collecting baseline trades + indicator snapshots (10 symbols)", flush=True)
    print("=" * w, flush=True)

    all_rows: list[tuple[LivePortfolioTrade, dict]] = []
    sym_counts: dict[str, int] = {}
    for sym in SYMBOLS:
        df_5m = load_candles(sym, cache_dir)
        if df_5m.empty:
            print(f"  SKIP {sym} (no cached data for {FROM}_{TO})", flush=True)
            continue
        sym_total = 0
        for tf in TIMEFRAMES:
            t0 = time.time()
            ctx_df = build_bars_for_tf(df_5m, CONTEXT_TF.get(tf, "4h"))
            bars = build_bars_for_tf(df_5m, tf)
            chunk = collect_trades_live_aligned(
                bars, ctx_df, sym, tf, snap_fn=lambda b, c: _snap(b, c),
            )
            all_rows.extend(chunk)
            sym_total += len(chunk)
            print(f"  {tf:>4s} {sym:<18s}: {len(chunk):>5d} trades  ({time.time()-t0:.1f}s)",
                  flush=True)
        sym_counts[sym] = sym_total

    all_rows.sort(key=lambda x: x[0].entry_time)
    n_base = len(all_rows)
    print(f"\n  TOTAL baseline trades: {n_base:,}", flush=True)
    for sym in SYMBOLS:
        print(f"    {sym:<18s}: {sym_counts.get(sym, 0):>5}", flush=True)

    # ---- Grid sweep across windows ----
    print(f"\n{'='*w}", flush=True)
    print("PHASE 2: Grid sweep across all windows, filters, risk, maxConc, volFloor", flush=True)
    print("=" * w, flush=True)

    results: list[dict] = []
    t_grid = time.time()
    done = 0

    for win_name, (ws, we) in WINDOWS.items():
        win_rows = _filter_trades_by_window(all_rows, ws, we) if win_name != "FULL" else all_rows
        print(f"\n  Window: {win_name} ({ws} -> {we}) - {len(win_rows):,} trades", flush=True)

        for label, pred in FILTER_SCENARIOS:
            for vf in VOL_FLOOR_GRID:
                filt = [t for t, snap in win_rows
                        if pred(snap, t.direction) and vol_floor(snap, t.direction, vf)]
                for mc in MAX_CONCURRENT_GRID:
                    for rp in RISK_GRID:
                        r = run_portfolio_live_aligned(
                            filt,
                            start_equity=START_EQUITY,
                            risk_pct=rp,
                            max_concurrent=mc,
                        )
                        results.append({
                            "window": win_name,
                            "filter": label,
                            "vol_floor": vf,
                            "max_conc": mc,
                            "risk_pct": rp,
                            "kept": len(filt),
                            "final_eq": r["final_eq"],
                            "pnl": r["pnl"],
                            "dd": r["max_dd_pct"],
                            "dd_usd": r["max_dd_usd"],
                            "trades": r["trades"],
                            "wr": r["wr"],
                            "wins": r["wins"],
                            "fees": r["total_fees"],
                            "funding": r["total_funding"],
                            "tp": r["tp"],
                            "sl": r["sl"],
                            "trail": r["trail"],
                            "timeout": r["timeout"],
                            "min_eq": r["min_equity"],
                            "skipped": r["skipped"],
                        })
                        done += 1
                        if done % 500 == 0:
                            print(f"    ... {done:,}/{total_configs:,} configs evaluated", flush=True)

    elapsed = time.time() - t_grid
    print(f"\n  Total: {len(results):,} configs in {elapsed:.1f}s", flush=True)

    # ---- Analysis: Find configs that survive bear windows ----
    print(f"\n{'='*w}", flush=True)
    print("PHASE 3: BEAR SURVIVAL ANALYSIS", flush=True)
    print("=" * w, flush=True)

    config_key = lambda r: (r["filter"], r["vol_floor"], r["max_conc"], r["risk_pct"])

    by_config: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in results:
        by_config[config_key(r)][r["window"]] = r

    scored: list[dict] = []
    for cfg, windows_data in by_config.items():
        if "FULL" not in windows_data:
            continue
        full = windows_data["FULL"]
        bear23 = windows_data.get("BEAR-2023")
        bear_late = windows_data.get("BEAR-LATE")

        bear23_pnl = bear23["pnl"] if bear23 else 0
        bear_late_pnl = bear_late["pnl"] if bear_late else 0
        bear23_dd = bear23["dd"] if bear23 else 0
        bear_late_dd = bear_late["dd"] if bear_late else 0

        worst_bear_pnl = min(bear23_pnl, bear_late_pnl)
        avg_bear_pnl = (bear23_pnl + bear_late_pnl) / 2

        bear_survival_score = (
            full["pnl"] * 0.4
            + worst_bear_pnl * 100 * 0.3
            + avg_bear_pnl * 100 * 0.2
            - max(bear23_dd, bear_late_dd) * full["pnl"] * 0.1
        )

        scored.append({
            "filter": cfg[0],
            "vol_floor": cfg[1],
            "max_conc": cfg[2],
            "risk_pct": cfg[3],
            "full_pnl": full["pnl"],
            "full_dd": full["dd"],
            "full_eq": full["final_eq"],
            "full_trades": full["trades"],
            "full_wr": full["wr"],
            "bear23_pnl": bear23_pnl,
            "bear23_dd": bear23_dd,
            "bear_late_pnl": bear_late_pnl,
            "bear_late_dd": bear_late_dd,
            "worst_bear": worst_bear_pnl,
            "avg_bear": avg_bear_pnl,
            "score": bear_survival_score,
            "full_data": full,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    # ---- Top 40 table ----
    print(f"\nTOP 40 CONFIGS BY BEAR SURVIVAL SCORE", flush=True)
    print(f"(Score = 40% full PnL + 30% worst bear PnL*100 + 20% avg bear PnL*100 "
          f"- 10% worst bear DD * full PnL)\n", flush=True)
    hdr = (f"{'#':>3}  {'Filter':<28}  {'VF':>4}  {'MC':>3}  {'Risk':>5}  "
           f"{'FullPnL':>9}  {'FullDD':>6}  {'B23PnL':>8}  {'B23DD':>6}  "
           f"{'BLtPnL':>8}  {'BLtDD':>6}  {'WR':>5}  {'Tr':>5}  {'Score':>9}")
    print(hdr, flush=True)
    print("-" * w, flush=True)
    for i, x in enumerate(scored[:40], 1):
        print(
            f"{i:>3}  {x['filter']:<28}  {x['vol_floor']:>3.1f}  {x['max_conc']:>3}  "
            f"{x['risk_pct']:>4.1%}  ${x['full_pnl']:>+7,.0f}  {x['full_dd']:>5.1%}  "
            f"${x['bear23_pnl']:>+6,.0f}  {x['bear23_dd']:>5.1%}  "
            f"${x['bear_late_pnl']:>+6,.0f}  {x['bear_late_dd']:>5.1%}  "
            f"{x['full_wr']:>4.1f}%  {x['full_trades']:>5}  {x['score']:>+9,.0f}",
            flush=True,
        )

    # ---- Best bear survivors: positive in BOTH bear windows ----
    print(f"\n{'='*w}", flush=True)
    print("CONFIGS PROFITABLE IN BOTH BEAR WINDOWS (sorted by full PnL)", flush=True)
    print("=" * w, flush=True)

    both_positive = [x for x in scored if x["bear23_pnl"] > 0 and x["bear_late_pnl"] > 0]
    both_positive.sort(key=lambda x: x["full_pnl"], reverse=True)

    if both_positive:
        print(f"\n  Found {len(both_positive)} configs profitable in BOTH bear windows\n", flush=True)
        hdr2 = (f"{'#':>3}  {'Filter':<28}  {'VF':>4}  {'MC':>3}  {'Risk':>5}  "
                f"{'FullPnL':>9}  {'FullDD':>6}  {'B23PnL':>8}  {'BLtPnL':>8}  "
                f"{'WR':>5}  {'Tr':>5}")
        print(hdr2, flush=True)
        print("-" * w, flush=True)
        for i, x in enumerate(both_positive[:25], 1):
            print(
                f"{i:>3}  {x['filter']:<28}  {x['vol_floor']:>3.1f}  {x['max_conc']:>3}  "
                f"{x['risk_pct']:>4.1%}  ${x['full_pnl']:>+7,.0f}  {x['full_dd']:>5.1%}  "
                f"${x['bear23_pnl']:>+6,.0f}  ${x['bear_late_pnl']:>+6,.0f}  "
                f"{x['full_wr']:>4.1f}%  {x['full_trades']:>5}",
                flush=True,
            )
    else:
        print("\n  No configs are profitable in BOTH bear windows.", flush=True)
        print("  Showing top 15 with LEAST LOSS in worst bear window:\n", flush=True)
        least_loss = sorted(scored, key=lambda x: x["worst_bear"], reverse=True)[:15]
        for i, x in enumerate(least_loss, 1):
            print(
                f"{i:>3}  {x['filter']:<28}  VF={x['vol_floor']:.1f}  MC={x['max_conc']}  "
                f"Risk={x['risk_pct']:.1%}  FullPnL=${x['full_pnl']:+,.0f}  "
                f"B23=${x['bear23_pnl']:+,.0f}  BLt=${x['bear_late_pnl']:+,.0f}",
                flush=True,
            )

    # ---- Per-symbol contribution for the best bear-survival config ----
    best = scored[0]
    best_pred = None
    for label, pred in FILTER_SCENARIOS:
        if label == best["filter"]:
            best_pred = pred
            break

    if best_pred:
        print(f"\n{'='*w}", flush=True)
        print(f"PER-SYMBOL BREAKDOWN for best bear-survival config:", flush=True)
        print(f"  Filter: {best['filter']}  |  VolFloor: {best['vol_floor']}  |  "
              f"MaxConc: {best['max_conc']}  |  Risk: {best['risk_pct']:.2%}", flush=True)
        print(f"\n{'Symbol':<18}  {'Signals':>8}  {'Wins':>6}  {'WR':>6}  {'AvgRaw%':>8}", flush=True)
        print("-" * 55, flush=True)

        for win_name in ["FULL", "BEAR-2023", "BEAR-LATE"]:
            ws, we = WINDOWS[win_name]
            win_rows = _filter_trades_by_window(all_rows, ws, we) if win_name != "FULL" else all_rows
            sym_stats = _per_symbol_stats(win_rows, best_pred, best["vol_floor"])
            print(f"\n  --- {win_name} ({ws} -> {we}) ---", flush=True)
            for sym in SYMBOLS:
                if sym in sym_stats:
                    ss = sym_stats[sym]
                    print(f"  {sym:<18}  {ss['n']:>8}  {ss['wins']:>6}  "
                          f"{ss['wr']:>5.1f}%  {ss['avg_raw_pct']:>+7.2f}%", flush=True)
                else:
                    print(f"  {sym:<18}  {0:>8}  {0:>6}  {'N/A':>6}  {'N/A':>8}", flush=True)

    # ---- Monthly PnL for the best config (FULL period) ----
    if best_pred:
        best_filt = [t for t, snap in all_rows
                     if best_pred(snap, t.direction)
                     and vol_floor(snap, t.direction, best["vol_floor"])]
        monthly = _monthly_pnl(
            best_filt, best["risk_pct"], best["max_conc"],
            START_EQUITY, FROM,
        )
        if monthly:
            print(f"\n{'='*w}", flush=True)
            print(f"MONTHLY PnL for best config (FULL period)", flush=True)
            print(f"{'Month':<10}  {'PnL':>10}  {'%':>7}  {'EndEq':>10}  {'Trades':>7}", flush=True)
            print("-" * 50, flush=True)
            for m in monthly:
                print(f"{m['month']:<10}  ${m['pnl']:>+8,.0f}  {m['pct']:>+6.1f}%  "
                      f"${m['end_eq']:>8,.0f}  {m['trades']:>7}", flush=True)

    # ---- Window comparison for all filters at their best risk ----
    print(f"\n{'='*w}", flush=True)
    print("WINDOW COMPARISON: Best risk per filter (showing all windows side by side)", flush=True)
    print("=" * w, flush=True)

    filter_best: dict[str, dict] = {}
    for x in scored:
        fk = f"{x['filter']}|VF={x['vol_floor']}|MC={x['max_conc']}"
        if fk not in filter_best:
            filter_best[fk] = x

    hdr3 = (f"{'Filter + Params':<45}  {'Risk':>5}  "
            f"{'FullPnL':>9}  {'B23PnL':>8}  {'BLtPnL':>8}  {'Score':>9}")
    print(hdr3, flush=True)
    print("-" * w, flush=True)
    sorted_fb = sorted(filter_best.values(), key=lambda x: x["score"], reverse=True)
    for x in sorted_fb[:30]:
        fk = f"{x['filter']} VF={x['vol_floor']:.1f} MC={x['max_conc']}"
        print(
            f"{fk:<45}  {x['risk_pct']:>4.1%}  "
            f"${x['full_pnl']:>+7,.0f}  ${x['bear23_pnl']:>+6,.0f}  "
            f"${x['bear_late_pnl']:>+6,.0f}  {x['score']:>+9,.0f}",
            flush=True,
        )

    # ---- Large-cap vs altcoin analysis ----
    print(f"\n{'='*w}", flush=True)
    print("LARGE-CAP vs ALTCOIN ANALYSIS (best bear-survival filter)", flush=True)
    print("=" * w, flush=True)

    if best_pred:
        largecap = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
        altcoins = [s for s in SYMBOLS if s not in largecap]

        for group_name, group_syms in [("LARGE-CAP (BTC/ETH/XRP)", largecap),
                                        ("ALTCOINS (current 7)", altcoins)]:
            group_trades = [t for t, snap in all_rows
                           if t.symbol in group_syms
                           and best_pred(snap, t.direction)
                           and vol_floor(snap, t.direction, best["vol_floor"])]
            if not group_trades:
                print(f"\n  {group_name}: no trades", flush=True)
                continue

            for win_name in ["FULL", "BEAR-2023", "BEAR-LATE"]:
                ws, we = WINDOWS[win_name]
                if win_name == "FULL":
                    wt = group_trades
                else:
                    wst = pd.Timestamp(ws, tz="UTC")
                    wet = pd.Timestamp(we, tz="UTC")
                    wt = []
                    for t in group_trades:
                        et = t.entry_time
                        if et.tzinfo is None:
                            et = et.tz_localize("UTC")
                        if wst <= et <= wet:
                            wt.append(t)

                r = run_portfolio_live_aligned(
                    wt, start_equity=START_EQUITY, risk_pct=best["risk_pct"],
                    max_concurrent=best["max_conc"],
                )
                print(f"\n  {group_name} - {win_name}:", flush=True)
                print(f"    Trades: {r['trades']:,}  |  WR: {r['wr']:.1f}%  |  "
                      f"PnL: ${r['pnl']:+,.0f}  |  DD: {r['max_dd_pct']:.1%}  |  "
                      f"Final: ${r['final_eq']:,.0f}", flush=True)

    print(f"\n{'='*w}", flush=True)
    print("SIMULATION COMPLETE", flush=True)
    print("=" * w, flush=True)


if __name__ == "__main__":
    main()
