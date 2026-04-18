"""
STRATEGY COMPARISON SIMULATION

Runs each existing strategy through the bear market sim engine and compares
results against the baseline Sniper strategy.

Strategies tested:
  1. sniper          (baseline)
  2. bb_squeeze      (breakout)
  3. rsi_divergence  (momentum reversal)
  4. vwap_reversion  (mean reversion)
  5. multitf_scalp   (5m scalp with 15m trend)
  6. high_winrate    (trend-following pullback)
  7. volume_delta_liq (breakout with flow)
  8. Combined: sniper + best complementary strategies

Run:
    python scripts/strategy_sim.py
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

from src.backtest.live_aligned_portfolio import (
    LivePortfolioTrade,
    collect_trades_live_aligned,
    run_portfolio_live_aligned,
    synthesize_volume_delta,
)
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.base import BaseStrategy, _sf

START_EQUITY = 3_000.0
RISK_PCT = 0.025
MAX_CONCURRENT = 4

SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "DOGEUSDT", "OPUSDT", "BTCUSDT",
]
TIMEFRAMES = ["15m", "4h"]
FROM = "2023-01-01"
TO = "2026-04-03"

WINDOWS = {
    "FULL":      (FROM, TO),
    "BEAR-2023": ("2023-01-01", "2023-12-31"),
    "BEAR-LATE": ("2025-11-01", "2026-04-03"),
}

_CHOP_VOL_FLOOR = 1.2  # NEW-6: best filter from grid search (+2,445% in 3yr)

STRATEGIES_TO_TEST = [
    "sniper", "bb_squeeze", "rsi_divergence", "vwap_reversion",
    "multitf_scalp", "high_winrate", "volume_delta_liq",
]

# Strategies whose intrinsic trigger logic *aligns* with the trend-following
# anti-chop filter (ADX/DI/BBmid/vol).  Mean-reversion and squeeze
# strategies are intentionally exempt — applying that filter to them blocks
# every signal because they fire in the opposite regime.
_TREND_STRATEGIES = {"sniper", "multitf_scalp", "high_winrate", "volume_delta_liq"}


def _passes_chop_filter(bar, direction):
    """NEW-6 anti-chop: DI alignment + on right side of BB mid + vol >= 1.2x
    + no BB squeeze.  Identified as the highest-PnL filter in the 19-config
    grid search (best_filters_sim_results.txt: +2,445% over 3yr at 3% risk)."""
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
    if vr > 0 and vr < _CHOP_VOL_FLOOR:
        return False
    if bar.get("bb_squeeze") is True:
        return False
    return True


def collect_trades_for_strategy(
    strategy: BaseStrategy,
    trading_bars: pd.DataFrame,
    context_df: pd.DataFrame | None,
    symbol: str,
    tf: str,
    apply_chop_filter: bool = True,
) -> list[LivePortfolioTrade]:
    """
    Thin wrapper around the canonical
    ``collect_trades_live_aligned`` so every strategy benefits from
    partial-TP ladders, per-strategy cooldowns, fill-mode-aware
    entries, and funding accrued by 8h-fix count.
    """
    chop = (lambda b, d: _passes_chop_filter(b, d)) if apply_chop_filter else None
    rows = collect_trades_live_aligned(
        trading_bars=trading_bars,
        context_df=context_df,
        symbol=symbol,
        tf=tf,
        snap_fn=lambda _b, _c: {},
        strategy=strategy,
        chop_filter=chop,
    )
    return [t for t, _ in rows]


def load_candles(symbol, cache_dir):
    p = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    # CCXT-only OHLCV cache lacks tick flow; synthesize buy/sell from
    # candle shape so OFI-dependent strategies (bb_squeeze,
    # multitf_scalp, vwap_reversion) actually fire.  No-op when the
    # cache already has a real split.
    return synthesize_volume_delta(df)


def _filter_by_window(trades, ws, we):
    wst = pd.Timestamp(ws, tz="UTC")
    wet = pd.Timestamp(we, tz="UTC")
    out = []
    for t in trades:
        et = t.entry_time
        if et.tzinfo is None:
            et = et.tz_localize("UTC")
        if wst <= et <= wet:
            out.append(t)
    return out


def main():
    from src.strategies import STRATEGY_REGISTRY

    cache_dir = project_root / "data_cache"
    w = 140

    print("=" * w)
    print("STRATEGY COMPARISON SIMULATION")
    print("=" * w)
    print(f"  Filter:  NEW-6 (DI + BB mid + vol>=1.2x + no squeeze) trend-only")
    print(f"           [{', '.join(sorted(_TREND_STRATEGIES))}]")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  Risk:    {RISK_PCT:.1%}  |  MaxConc: {MAX_CONCURRENT}  |  Start: ${START_EQUITY:,.0f}")

    # PHASE 1: Precompute bars for all symbols/TFs (only once)
    print(f"\n{'='*w}")
    print("PHASE 1: Precomputing indicator bars (shared across all strategies)")
    print("=" * w)

    precomputed: dict[str, dict[str, tuple[pd.DataFrame, pd.DataFrame]]] = {}
    for sym in SYMBOLS:
        df_5m = load_candles(sym, cache_dir)
        if df_5m.empty:
            print(f"  SKIP {sym}", flush=True)
            continue
        precomputed[sym] = {}
        for tf in TIMEFRAMES:
            t0 = time.time()
            ctx_df = build_bars_for_tf(df_5m, CONTEXT_TF.get(tf, "4h"))
            bars = build_bars_for_tf(df_5m, tf)
            precomputed[sym][tf] = (bars, ctx_df)
            print(f"  {tf:>4s} {sym:<18s}: {len(bars):,} bars  ({time.time()-t0:.1f}s)", flush=True)

    # PHASE 2: Collect trades for each strategy using precomputed bars
    strategy_trades: dict[str, list[LivePortfolioTrade]] = {}

    for strat_name in STRATEGIES_TO_TEST:
        strat_cls = STRATEGY_REGISTRY[strat_name]
        all_trades: list[LivePortfolioTrade] = []
        t0_strat = time.time()

        print(f"\n  Collecting: {strat_name} ...", end="", flush=True)

        apply_chop = strat_name in _TREND_STRATEGIES
        for sym in precomputed:
            for tf in TIMEFRAMES:
                bars, ctx_df = precomputed[sym][tf]
                strat_instance = strat_cls()
                chunk = collect_trades_for_strategy(
                    strat_instance, bars, ctx_df, sym, tf,
                    apply_chop_filter=apply_chop,
                )
                all_trades.extend(chunk)

        all_trades.sort(key=lambda t: t.entry_time)
        strategy_trades[strat_name] = all_trades
        print(f" {len(all_trades):>4d} trades  ({time.time()-t0_strat:.1f}s)")

    # Run portfolio simulation for each strategy across each window
    print(f"\n{'='*w}")
    print("RESULTS BY STRATEGY AND WINDOW")
    print("=" * w)

    for win_name, (ws, we) in WINDOWS.items():
        print(f"\n{'='*w}")
        print(f"  WINDOW: {win_name} ({ws} -> {we})")
        print(f"{'='*w}")

        hdr = (f"  {'Strategy':<22}  {'PnL':>9}  {'Ret%':>7}  {'MaxDD':>7}  "
               f"{'WR':>6}  {'Tr':>5}  {'TP':>5}  {'SL':>5}  {'Trail':>5}  {'Fees':>8}")
        print(hdr)
        print("  " + "-" * (w - 4))

        for strat_name in STRATEGIES_TO_TEST:
            if win_name == "FULL":
                trades = strategy_trades[strat_name]
            else:
                trades = _filter_by_window(strategy_trades[strat_name], ws, we)

            if not trades:
                print(f"  {strat_name:<22}  {'--':>9}  {'--':>7}  {'--':>7}  {'--':>6}  {'0':>5}")
                continue

            r = run_portfolio_live_aligned(
                trades, start_equity=START_EQUITY, risk_pct=RISK_PCT,
                max_concurrent=MAX_CONCURRENT,
            )
            ret = r["pnl"] / START_EQUITY * 100
            print(
                f"  {strat_name:<22}  ${r['pnl']:>+7,.0f}  {ret:>+6.0f}%  "
                f"{r['max_dd_pct']:>6.1%}  {r['wr']:>5.1f}%  {r['trades']:>5}  "
                f"{r['tp']:>5}  {r['sl']:>5}  {r['trail']:>5}  ${r['total_fees']:>6,.0f}"
            )

    # Test combined portfolios
    print(f"\n{'='*w}")
    print("COMBINED STRATEGY PORTFOLIOS")
    print("=" * w)

    combos = [
        ("sniper + high_winrate", ["sniper", "high_winrate"]),
        ("sniper + bb_squeeze", ["sniper", "bb_squeeze"]),
        ("sniper + multitf_scalp", ["sniper", "multitf_scalp"]),
        ("sniper + vwap_reversion", ["sniper", "vwap_reversion"]),
        ("sniper + rsi_divergence", ["sniper", "rsi_divergence"]),
        ("sniper + vol_delta", ["sniper", "volume_delta_liq"]),
        ("sniper + hw + bb", ["sniper", "high_winrate", "bb_squeeze"]),
        ("sniper + hw + vwap", ["sniper", "high_winrate", "vwap_reversion"]),
        ("all 7 strategies", STRATEGIES_TO_TEST),
    ]

    for win_name, (ws, we) in WINDOWS.items():
        print(f"\n  WINDOW: {win_name}")

        hdr = (f"  {'Combo':<28}  {'PnL':>9}  {'Ret%':>7}  {'MaxDD':>7}  "
               f"{'WR':>6}  {'Tr':>5}  {'TP':>5}  {'SL':>5}  {'Trail':>5}")
        print(hdr)
        print("  " + "-" * (w - 4))

        for label, strat_names in combos:
            merged = []
            for sn in strat_names:
                if win_name == "FULL":
                    merged.extend(strategy_trades[sn])
                else:
                    merged.extend(_filter_by_window(strategy_trades[sn], ws, we))
            merged.sort(key=lambda t: t.entry_time)

            if not merged:
                print(f"  {label:<28}  {'--':>9}")
                continue

            r = run_portfolio_live_aligned(
                merged, start_equity=START_EQUITY, risk_pct=RISK_PCT,
                max_concurrent=MAX_CONCURRENT,
            )
            ret = r["pnl"] / START_EQUITY * 100
            print(
                f"  {label:<28}  ${r['pnl']:>+7,.0f}  {ret:>+6.0f}%  "
                f"{r['max_dd_pct']:>6.1%}  {r['wr']:>5.1f}%  {r['trades']:>5}  "
                f"{r['tp']:>5}  {r['sl']:>5}  {r['trail']:>5}"
            )

    print(f"\n{'='*w}")
    print("SIMULATION COMPLETE")
    print("=" * w)


if __name__ == "__main__":
    main()
