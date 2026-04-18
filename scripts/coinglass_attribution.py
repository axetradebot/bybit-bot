"""
CoinGlass liquidation-imbalance attribution analysis.

Question: does the ``cg_liq_imb`` feature carry alpha for the Sniper
strategy?  If yes, an upgrade from Hobbyist (4h) to Startup (30m) or
Standard (1m) tier may be worth the spend.  If not, we save the money.

Method
------
1. Fetch ~180 days of 4h aggregated liquidation history from CoinGlass
   for every production symbol (Hobbyist limit; the same data we already
   sync into ``coinglass_liquidation_bars``).
2. Replay the *exact* live-aligned Sniper backtest used by
   ``portfolio_sim.py`` over the same window.
3. For every trade, look up the closest preceding 4h liquidation bucket
   and compute ``cg_liq_imb = (short_liq - long_liq) / (short_liq + long_liq)``
   — identical to the production formula in ``compute_all.py``.
4. Bucket trades two ways:
     a) Absolute imbalance: very-short-bias / mild-short / neutral /
        mild-long / very-long-bias.
     b) Direction-conditional: TAILWIND / NEUTRAL / HEADWIND
        ("tailwind" = imbalance pushing in the trade's favour, e.g.
        a long when shorts are getting liquidated).
5. Per bucket compute sample size, win rate, mean/median R-multiple,
   total R, and a t-test vs the global mean.  R-multiple is used as
   the outcome metric so the analysis is risk-percent-agnostic.
6. Decision rule:
     IMPLEMENT if any bucket has |t| > 1.96 (95% conf), n >= 50, AND
     mean-R differs from global by >= 0.10R.
     SHELVE otherwise — current 4h Hobbyist data carries no measurable edge.

Usage
-----
    python scripts/coinglass_attribution.py
    python scripts/coinglass_attribution.py --refresh-coinglass  # re-fetch
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.backtest.live_aligned_portfolio import (
    collect_trades_live_aligned,
    synthesize_volume_delta,
)
from src.data.coinglass_liquidation import CoinGlassClient
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.base import _sf

# Mirror portfolio_sim.py exactly so the trades we attribute are the
# same trades the live bot would take.
SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT",
    "1000PEPEUSDT", "DOGEUSDT", "OPUSDT", "BTCUSDT", "XRPUSDT",
]
TIMEFRAMES = ["15m", "4h"]
FROM = "2023-01-01"
TO = "2026-04-03"

CG_INTERVAL = "4h"  # max granularity on Hobbyist
CG_HISTORY_LIMIT = 1000  # API max — gives ~166 days at 4h
CG_CACHE = project_root / "data_cache" / "coinglass_4h_liq.parquet"

_CHOP_VOL_FLOOR = 1.2


def _chop_filter(bar, direction: str) -> bool:
    p, m = _sf(bar.get("plus_di")), _sf(bar.get("minus_di"))
    if p > 0 or m > 0:
        if direction == "long" and p <= m:
            return False
        if direction == "short" and m <= p:
            return False
    bb_mid = _sf(bar.get("bb_mid"))
    close = _sf(bar.get("close"))
    if bb_mid > 0 and close > 0:
        if direction == "long" and close <= bb_mid:
            return False
        if direction == "short" and close >= bb_mid:
            return False
    vr = _sf(bar.get("volume_ratio"))
    if vr > 0 and vr < _CHOP_VOL_FLOOR:
        return False
    sq = bar.get("bb_squeeze")
    if sq is True:
        return False
    return True


def _make_sniper():
    """Mirror portfolio_sim.py: ladder disabled, cooldown=6."""
    from src.strategies.strategy_sniper import SniperStrategy
    s = SniperStrategy()
    s.default_tp_ladder = ()
    s.move_be_on_tp1 = False
    s.cooldown_bars = 6
    return s


# ────────────────────────────────────────────────────────────────────────
# Step 1 — fetch CoinGlass liquidation bars
# ────────────────────────────────────────────────────────────────────────

def fetch_coinglass_history(refresh: bool) -> pd.DataFrame:
    """Pull 4h aggregated liq history for every SYMBOL.

    Caches to ``data_cache/coinglass_4h_liq.parquet`` so re-runs of the
    attribution don't burn API quota.
    """
    if not refresh and CG_CACHE.exists():
        df = pd.read_parquet(CG_CACHE)
        print(f"  Loaded {len(df):,} CoinGlass bars from cache "
              f"({CG_CACHE.name})", flush=True)
        return df

    client = CoinGlassClient()
    rows: list[dict] = []
    for sym in SYMBOLS:
        t0 = time.time()
        try:
            data = client.aggregated_liquidation_history(
                sym, interval=CG_INTERVAL, limit=CG_HISTORY_LIMIT,
            )
        except Exception as exc:
            print(f"  {sym}: FAILED — {exc}", flush=True)
            continue
        for r in data:
            ts_ms = r.get("time")
            if ts_ms is None:
                continue
            rows.append({
                "symbol": sym,
                "bucket_time": pd.Timestamp(int(ts_ms), unit="ms", tz="UTC"),
                "long_liq_usd": float(
                    r.get("aggregated_long_liquidation_usd") or 0.0
                ),
                "short_liq_usd": float(
                    r.get("aggregated_short_liquidation_usd") or 0.0
                ),
            })
        elapsed = time.time() - t0
        print(f"  {sym}: {len(data)} bars  ({elapsed:.1f}s)", flush=True)

    if not rows:
        raise RuntimeError("CoinGlass returned no data — check API key / quota")

    df = pd.DataFrame(rows)
    df["cg_liq_imb"] = (
        (df["short_liq_usd"] - df["long_liq_usd"])
        / (df["short_liq_usd"] + df["long_liq_usd"] + 1e-9)
    )
    df = df.sort_values(["symbol", "bucket_time"]).reset_index(drop=True)
    CG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CG_CACHE)
    print(f"\n  Cached {len(df):,} bars → {CG_CACHE.name}", flush=True)
    return df


# ────────────────────────────────────────────────────────────────────────
# Step 2 — replay Sniper trades (live-aligned)
# ────────────────────────────────────────────────────────────────────────

def load_candles(symbol: str) -> pd.DataFrame:
    cache_dir = project_root / "data_cache"
    cache_path = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    if not cache_path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(cache_path)
    return synthesize_volume_delta(df)


def collect_all_trades() -> pd.DataFrame:
    rows: list[dict] = []
    for sym in SYMBOLS:
        df_5m = load_candles(sym)
        if df_5m.empty:
            print(f"  {sym}: no candle cache, skipping", flush=True)
            continue
        for tf in TIMEFRAMES:
            t0 = time.time()
            context_tf = CONTEXT_TF.get(tf, "4h")
            ctx_df = build_bars_for_tf(df_5m, context_tf)
            trading_bars = build_bars_for_tf(df_5m, tf)
            trades = collect_trades_live_aligned(
                trading_bars, ctx_df, sym, tf,
                snap_fn=lambda _b, _c: {},
                strategy=_make_sniper(),
                chop_filter=_chop_filter,
            )
            for trade, _snap in trades:
                sl_pct = (
                    trade.sl_distance / trade.entry_price
                    if trade.entry_price > 0 else float("nan")
                )
                if not (sl_pct > 0):
                    continue
                # Total return % including partial-TP contribution.
                effective_pct = (
                    trade.partial_pnl_pct
                    + trade.raw_pct * trade.remaining_fraction
                )
                r_mult = effective_pct / sl_pct
                rows.append({
                    "symbol": sym,
                    "tf": tf,
                    "direction": trade.direction,
                    "entry_time": trade.entry_time,
                    "exit_time": trade.exit_time,
                    "raw_pct": trade.raw_pct,
                    "effective_pct": effective_pct,
                    "sl_pct": sl_pct,
                    "r_multiple": r_mult,
                    "exit_reason": trade.exit_reason,
                    "bars_held": trade.bars_held,
                })
            print(f"  {tf} {sym}: {len(trades)} trades  "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return pd.DataFrame(rows)


# ────────────────────────────────────────────────────────────────────────
# Step 3 — attach cg_liq_imb to each trade (backward merge)
# ────────────────────────────────────────────────────────────────────────

def attach_cg_imb(trades: pd.DataFrame, cg: pd.DataFrame) -> pd.DataFrame:
    """For every trade, find the closest preceding 4h bucket per symbol."""
    if trades.empty:
        return trades

    trades = trades.copy()
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    cg = cg.copy()
    cg["bucket_time"] = pd.to_datetime(cg["bucket_time"], utc=True)

    out_chunks = []
    for sym, g in trades.groupby("symbol", sort=False):
        sym_cg = cg[cg["symbol"] == sym][["bucket_time", "cg_liq_imb"]]
        if sym_cg.empty:
            g = g.copy()
            g["cg_liq_imb"] = np.nan
            out_chunks.append(g)
            continue
        sym_cg = sym_cg.sort_values("bucket_time")
        g = g.sort_values("entry_time")
        merged = pd.merge_asof(
            g, sym_cg,
            left_on="entry_time", right_on="bucket_time",
            direction="backward",
            tolerance=pd.Timedelta(hours=8),
        )
        merged = merged.drop(columns=["bucket_time"], errors="ignore")
        out_chunks.append(merged)
    return pd.concat(out_chunks, ignore_index=True)


# ────────────────────────────────────────────────────────────────────────
# Step 4 — bucket + statistical analysis
# ────────────────────────────────────────────────────────────────────────

def _welch_t(x: np.ndarray, mu: float) -> tuple[float, int]:
    """One-sample Welch t-stat of x vs population mean mu."""
    n = len(x)
    if n < 2:
        return 0.0, n
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    if sd == 0.0:
        return 0.0, n
    return (mean - mu) / (sd / math.sqrt(n)), n


def bucket_imbalance(imb: float) -> str:
    if not np.isfinite(imb):
        return "no_data"
    if imb < -0.3:
        return "very_long_bias_(-1..-0.3)"
    if imb < -0.1:
        return "mild_long_bias_(-0.3..-0.1)"
    if imb <= 0.1:
        return "neutral_(-0.1..0.1)"
    if imb <= 0.3:
        return "mild_short_bias_(0.1..0.3)"
    return "very_short_bias_(0.3..1)"


def directional_bucket(imb: float, direction: str) -> str:
    """TAILWIND if imbalance favours the trade direction.

    For a *long*: positive imbalance = short-side liquidations (bullish
    tailwind).  For a *short*: negative imbalance = long-side liqs
    (bearish tailwind).
    """
    if not np.isfinite(imb):
        return "no_data"
    aligned = imb if direction == "long" else -imb
    if aligned >= 0.3:
        return "TAILWIND_(>=0.3)"
    if aligned >= 0.1:
        return "MILD_TAIL_(0.1..0.3)"
    if aligned > -0.1:
        return "NEUTRAL_(|x|<0.1)"
    if aligned > -0.3:
        return "MILD_HEAD_(-0.3..-0.1)"
    return "HEADWIND_(<=-0.3)"


def summarise_bucket(name: str, sub: pd.DataFrame, mu: float) -> dict:
    r = sub["r_multiple"].to_numpy(dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    if n == 0:
        return {"bucket": name, "n": 0}
    wins = (r > 0).sum()
    wr = wins / n
    mean_r = float(np.mean(r))
    median_r = float(np.median(r))
    sum_r = float(np.sum(r))
    t_stat, _ = _welch_t(r, mu)
    return {
        "bucket": name,
        "n": int(n),
        "win_rate": round(wr * 100, 1),
        "mean_R": round(mean_r, 3),
        "median_R": round(median_r, 3),
        "sum_R": round(sum_r, 1),
        "t_vs_global": round(t_stat, 2),
        "delta_R_vs_global": round(mean_r - mu, 3),
    }


def analyse(trades: pd.DataFrame) -> dict:
    have = trades[trades["cg_liq_imb"].notna()].copy()
    no_data = len(trades) - len(have)
    print(f"\n  {len(have):,} trades within CoinGlass coverage  "
          f"({no_data:,} pre-coverage trades excluded)", flush=True)
    if have.empty:
        return {"verdict": "NO_DATA"}

    global_mu = float(have["r_multiple"].mean())

    have["bucket_abs"] = have["cg_liq_imb"].apply(bucket_imbalance)
    have["bucket_dir"] = have.apply(
        lambda r: directional_bucket(r["cg_liq_imb"], r["direction"]),
        axis=1,
    )

    abs_summary: list[dict] = []
    for name in [
        "very_long_bias_(-1..-0.3)",
        "mild_long_bias_(-0.3..-0.1)",
        "neutral_(-0.1..0.1)",
        "mild_short_bias_(0.1..0.3)",
        "very_short_bias_(0.3..1)",
    ]:
        abs_summary.append(summarise_bucket(
            name, have[have["bucket_abs"] == name], global_mu,
        ))

    dir_summary: list[dict] = []
    for name in [
        "TAILWIND_(>=0.3)",
        "MILD_TAIL_(0.1..0.3)",
        "NEUTRAL_(|x|<0.1)",
        "MILD_HEAD_(-0.3..-0.1)",
        "HEADWIND_(<=-0.3)",
    ]:
        dir_summary.append(summarise_bucket(
            name, have[have["bucket_dir"] == name], global_mu,
        ))

    long_summary = summarise_bucket(
        "ALL_LONGS", have[have["direction"] == "long"], global_mu,
    )
    short_summary = summarise_bucket(
        "ALL_SHORTS", have[have["direction"] == "short"], global_mu,
    )

    return {
        "global_mean_R": round(global_mu, 3),
        "global_n": int(len(have)),
        "absolute_imbalance": abs_summary,
        "direction_conditional": dir_summary,
        "longs_overall": long_summary,
        "shorts_overall": short_summary,
        "coverage_window": {
            "earliest_trade": str(have["entry_time"].min()),
            "latest_trade": str(have["entry_time"].max()),
        },
    }


def simulate_gate(trades: pd.DataFrame, threshold: float) -> dict:
    """What does the bot's PnL look like if we BLOCK the HEADWIND bucket?

    Mirrors ``RiskManager._cg_headwind_gate`` exactly:
        long  + cg_liq_imb <= -threshold → blocked
        short + cg_liq_imb >= +threshold → blocked
    Trades with no cg_liq_imb data fail-open (kept).
    """
    have = trades[trades["cg_liq_imb"].notna()].copy()
    if have.empty:
        return {"threshold": threshold}

    headwind = (
        ((have["direction"] == "long") & (have["cg_liq_imb"] <= -threshold))
        | ((have["direction"] == "short") & (have["cg_liq_imb"] >= threshold))
    )
    kept = have[~headwind]
    blocked = have[headwind]

    def _stats(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"n": 0}
        r = df["r_multiple"].to_numpy(dtype=np.float64)
        r = r[np.isfinite(r)]
        if len(r) == 0:
            return {"n": 0}
        return {
            "n": int(len(r)),
            "win_rate": round((r > 0).mean() * 100, 1),
            "mean_R": round(float(r.mean()), 3),
            "sum_R": round(float(r.sum()), 1),
        }

    before = _stats(have)
    after = _stats(kept)
    blocked_stats = _stats(blocked)

    pnl_uplift_pct = None
    if before.get("sum_R", 0):
        pnl_uplift_pct = round(
            (after["sum_R"] - before["sum_R"]) / abs(before["sum_R"]) * 100,
            1,
        )

    return {
        "threshold": threshold,
        "before_gate": before,
        "after_gate": after,
        "blocked_trades": blocked_stats,
        "trade_count_change_pct": round(
            (after["n"] - before["n"]) / before["n"] * 100, 1,
        ),
        "pnl_uplift_pct": pnl_uplift_pct,
    }


def make_verdict(analysis: dict) -> str:
    """Decision rule from the docstring."""
    if analysis.get("verdict") == "NO_DATA":
        return "NO_DATA"
    candidates = (
        analysis["absolute_imbalance"]
        + analysis["direction_conditional"]
    )
    edges = [
        b for b in candidates
        if b.get("n", 0) >= 50
        and abs(b.get("t_vs_global", 0.0)) >= 1.96
        and abs(b.get("delta_R_vs_global", 0.0)) >= 0.10
    ]
    if not edges:
        return "SHELVE"
    return "IMPLEMENT"


# ────────────────────────────────────────────────────────────────────────
# Driver
# ────────────────────────────────────────────────────────────────────────

def _print_table(title: str, rows: list[dict]):
    print(f"\n  {title}", flush=True)
    print(f"  {'-' * 110}", flush=True)
    print(f"  {'bucket':<32s}  {'n':>6s}  {'WR':>6s}  "
          f"{'meanR':>7s}  {'medR':>7s}  {'sumR':>8s}  "
          f"{'tvsG':>6s}  {'dR':>7s}", flush=True)
    print(f"  {'-' * 110}", flush=True)
    for r in rows:
        if r.get("n", 0) == 0:
            print(f"  {r['bucket']:<32s}  {'0':>6s}  {'-':>6s}  "
                  f"{'-':>7s}  {'-':>7s}  {'-':>8s}  "
                  f"{'-':>6s}  {'-':>7s}", flush=True)
            continue
        print(f"  {r['bucket']:<32s}  {r['n']:>6d}  "
              f"{r['win_rate']:>5.1f}%  "
              f"{r['mean_R']:>+7.3f}  {r['median_R']:>+7.3f}  "
              f"{r['sum_R']:>+8.1f}  {r['t_vs_global']:>+6.2f}  "
              f"{r['delta_R_vs_global']:>+7.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-coinglass", action="store_true",
        help="Re-fetch liquidation history from CoinGlass (uses API quota).",
    )
    args = parser.parse_args()

    print("=" * 78, flush=True)
    print("CoinGlass Liquidation-Imbalance Attribution Analysis", flush=True)
    print("=" * 78, flush=True)

    print("\n-- Step 1: load CoinGlass 4h liquidation history --", flush=True)
    cg = fetch_coinglass_history(refresh=args.refresh_coinglass)
    print(f"  Coverage per symbol:", flush=True)
    cov = (
        cg.groupby("symbol")["bucket_time"]
        .agg(["min", "max", "count"]).reset_index()
    )
    for _, row in cov.iterrows():
        print(f"    {row['symbol']:<14s}  "
              f"{str(row['min'])[:19]} → {str(row['max'])[:19]}  "
              f"({int(row['count'])} bars)", flush=True)

    print("\n-- Step 2: replay live-aligned Sniper trades --", flush=True)
    trades = collect_all_trades()
    print(f"\n  Total replayed trades: {len(trades):,}", flush=True)
    if trades.empty:
        print("\n  No trades produced — abort", flush=True)
        return

    print("\n-- Step 3: attach cg_liq_imb to each trade --", flush=True)
    trades = attach_cg_imb(trades, cg)

    print("\n-- Step 4: bucketed analysis --", flush=True)
    analysis = analyse(trades)
    print(f"\n  Global mean-R: {analysis['global_mean_R']:+.3f}  "
          f"(n={analysis['global_n']:,})", flush=True)
    print(f"  Window: {analysis['coverage_window']['earliest_trade'][:10]} "
          f"→ {analysis['coverage_window']['latest_trade'][:10]}", flush=True)

    _print_table("ABSOLUTE IMBALANCE BUCKETS",
                 analysis["absolute_imbalance"])
    _print_table("DIRECTION-CONDITIONAL BUCKETS (alpha test!)",
                 analysis["direction_conditional"])
    _print_table("BASELINE BY DIRECTION",
                 [analysis["longs_overall"], analysis["shorts_overall"]])

    print("\n  GATE-IMPACT SIMULATION (threshold sweep)", flush=True)
    print(f"  {'-' * 110}", flush=True)
    print(f"  {'thr':>5s}  {'kept_n':>7s}  {'kept_WR':>8s}  "
          f"{'kept_meanR':>11s}  {'kept_sumR':>10s}  "
          f"{'blocked_n':>10s}  {'tradeΔ%':>8s}  {'PnLΔ%':>8s}",
          flush=True)
    print(f"  {'-' * 110}", flush=True)
    sims = []
    for thr in (0.20, 0.30, 0.40, 0.50):
        s = simulate_gate(trades, threshold=thr)
        sims.append(s)
        b = s.get("after_gate", {})
        bl = s.get("blocked_trades", {})
        print(f"  {thr:>5.2f}  {b.get('n', 0):>7d}  "
              f"{b.get('win_rate', 0):>7.1f}%  "
              f"{b.get('mean_R', 0):>+11.3f}  "
              f"{b.get('sum_R', 0):>+10.1f}  "
              f"{bl.get('n', 0):>10d}  "
              f"{s.get('trade_count_change_pct', 0):>+7.1f}%  "
              f"{s.get('pnl_uplift_pct', 0):>+7.1f}%", flush=True)

    verdict = make_verdict(analysis)

    print("\n" + "=" * 78, flush=True)
    print(f"VERDICT: {verdict}", flush=True)
    print("=" * 78, flush=True)
    if verdict == "IMPLEMENT":
        print("  At least one bucket has n>=50, |t|>=1.96, |dR|>=0.10.",
              flush=True)
        print("  → Build cg_liq_imb gate; revisit Startup-tier upgrade.",
              flush=True)
    elif verdict == "SHELVE":
        print("  No bucket meets the n>=50 + |t|>=1.96 + |dR|>=0.10 bar.",
              flush=True)
        print("  → 4h CoinGlass data has no measurable edge for Sniper.",
              flush=True)
        print("  → Stay on Hobbyist ($29/mo); skip Startup ($600/yr save).",
              flush=True)

    out = project_root / "reports" / "coinglass_attribution.json"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        json.dump(
            {
                "analysis": analysis,
                "gate_simulation": sims,
                "verdict": verdict,
            },
            f, indent=2,
        )
    print(f"\n  Full results → {out.relative_to(project_root)}",
          flush=True)


if __name__ == "__main__":
    main()
