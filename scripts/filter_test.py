"""
Test ADX and BB-width filters on the Sniper strategy.

Collects all baseline signals in one pass, then replays with
different filter thresholds to compare PnL, win rate, and trade count.

Usage:
    python scripts/filter_test.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.strategy_sniper import SniperStrategy
from src.strategies.base import _sf

MAKER_FEE = 0.0002
TAKER_FEE = 0.00055
SLIPPAGE_BPS = 0.0003
ROUND_TRIP_COST = MAKER_FEE + TAKER_FEE + SLIPPAGE_BPS
NOTIONAL_CAP_FRAC = 0.25
COOLDOWN_BARS = 6
EQUITY = 3000.0
RISK_PCT = 0.02
LEVERAGE = 20
TRAIL_ACTIVATE = 0.10
TRAIL_OFFSET = 0.03

SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT",
    "1000PEPEUSDT", "DOGEUSDT", "OPUSDT", "SUIUSDT",
]
TIMEFRAMES = ["15m", "4h"]
FROM = "2023-01-01"
TO = "2026-04-03"


@dataclass
class Signal:
    bar_idx: int
    direction: str
    entry_price: float
    sl_distance: float
    tp_distance: float
    adx: float
    bb_width: float
    bb_squeeze: bool
    symbol: str
    tf: str


def collect_signals(trading_bars, context_df, symbol, tf):
    """Single pass: generate all sniper signals with indicator metadata."""
    sniper = SniperStrategy()
    signals = []

    ctx_ts, ctx_rows = [], []
    if context_df is not None and not context_df.empty:
        for _, row in context_df.iterrows():
            ctx_ts.append(pd.Timestamp(row["timestamp"]))
            ctx_rows.append(row)

    def get_ctx(ts):
        if not ctx_ts:
            return pd.Series(dtype=object)
        idx = 0
        for i, t in enumerate(ctx_ts):
            if t <= ts:
                idx = i
            else:
                break
        return ctx_rows[idx]

    for i in range(len(trading_bars)):
        bar = trading_bars.iloc[i]
        ctx = get_ctx(pd.Timestamp(bar.get("timestamp")))
        sig = sniper.generate_signal(
            symbol=symbol, indicators_5m=bar,
            indicators_15m=ctx, funding_rate=0.0,
            liq_volume_1h=0.0,
        )
        if sig is not None and sig.direction != "flat":
            sl_dist = abs(sig.entry_price - sig.stop_loss)
            tp_dist = abs(sig.entry_price - sig.take_profit)
            adx = _sf(bar.get("adx_14"))
            bb_w = _sf(bar.get("bb_width"))
            bb_sq = bool(bar.get("bb_squeeze", False))
            signals.append(Signal(
                bar_idx=i, direction=sig.direction,
                entry_price=sig.entry_price,
                sl_distance=sl_dist, tp_distance=tp_dist,
                adx=adx, bb_width=bb_w, bb_squeeze=bb_sq,
                symbol=symbol, tf=tf,
            ))
    return signals


def replay(signals: list[Signal], highs, lows, closes, n_bars) -> dict:
    """Replay filtered signals with full position sizing + trailing stop."""
    equity = EQUITY
    trades = 0
    wins = 0
    total_pnl = 0.0
    tp_count = 0
    sl_count = 0
    trail_count = 0
    last_exit_idx = -COOLDOWN_BARS - 1

    for sig in signals:
        if sig.bar_idx <= last_exit_idx + COOLDOWN_BARS:
            continue

        entry = sig.entry_price
        sl_dist = sig.sl_distance
        tp_dist = sig.tp_distance

        # Position sizing matching live
        if sl_dist <= 0 or entry <= 0:
            continue
        risk_amount = equity * RISK_PCT
        qty = risk_amount / (sl_dist + entry * ROUND_TRIP_COST)
        notional = qty * entry
        max_notional = equity * LEVERAGE * NOTIONAL_CAP_FRAC
        if notional > max_notional:
            notional = max_notional
            qty = notional / entry

        entry_fee = notional * MAKER_FEE

        if sig.direction == "long":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        # Scan for exit with trailing stop
        trail_active = False
        hwm = 0.0
        current_sl = sl
        exit_price = None
        exit_reason = None

        for j in range(sig.bar_idx + 1, n_bars):
            bh = highs[j]
            bl = lows[j]

            # Trailing stop logic
            if sig.direction == "long":
                gain = (bh - entry) / entry
                if gain >= TRAIL_ACTIVATE:
                    trail_active = True
                if trail_active:
                    if bh > hwm:
                        hwm = bh
                    new_sl = hwm * (1 - TRAIL_OFFSET)
                    if new_sl > current_sl:
                        current_sl = new_sl
                sl_hit = bl <= current_sl
                tp_hit = bh >= tp
            else:
                gain = (entry - bl) / entry
                if gain >= TRAIL_ACTIVATE:
                    trail_active = True
                if trail_active:
                    if hwm == 0 or bl < hwm:
                        hwm = bl
                    new_sl = hwm * (1 + TRAIL_OFFSET)
                    if new_sl < current_sl:
                        current_sl = new_sl
                sl_hit = bh >= current_sl
                tp_hit = bl <= tp

            if sl_hit:
                exit_price = current_sl
                exit_reason = "trail" if trail_active else "sl"
                last_exit_idx = j
                break
            elif tp_hit:
                exit_price = tp
                exit_reason = "tp"
                last_exit_idx = j
                break

        if exit_price is None:
            exit_price = closes[-1]
            exit_reason = "timeout"
            last_exit_idx = n_bars

        if sig.direction == "long":
            raw_pct = (exit_price - entry) / entry
        else:
            raw_pct = (entry - exit_price) / entry

        exit_fee = notional * TAKER_FEE
        pnl = notional * raw_pct - entry_fee - exit_fee
        equity += pnl
        total_pnl += pnl
        trades += 1
        if pnl > 0:
            wins += 1
        if exit_reason == "tp":
            tp_count += 1
        elif exit_reason == "trail":
            trail_count += 1
        elif exit_reason == "sl":
            sl_count += 1

    wr = wins / trades * 100 if trades else 0
    return {
        "trades": trades, "wins": wins, "wr": wr,
        "pnl": total_pnl, "final_eq": equity,
        "tp": tp_count, "sl": sl_count, "trail": trail_count,
    }


def load_candles(symbol, cache_dir):
    cache_path = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    print(f"  [WARN] No cache for {symbol}, skipping", flush=True)
    return pd.DataFrame()


def main():
    cache_dir = project_root / "data_cache"

    print("=" * 80, flush=True)
    print("ADX & BB-WIDTH FILTER TEST", flush=True)
    print("=" * 80, flush=True)

    # ADX thresholds to test
    adx_thresholds = [0, 15, 18, 20, 22, 25, 30]
    # BB width percentile thresholds (min bb_width to trade)
    bb_width_thresholds = [0, 0.01, 0.02, 0.03, 0.04, 0.05]

    # Load data and collect all signals
    print("\n-- Loading data & collecting signals --", flush=True)
    all_signals: list[Signal] = []
    bar_data: dict[tuple[str, str], tuple] = {}

    for sym in SYMBOLS:
        df_5m = load_candles(sym, cache_dir)
        if df_5m.empty:
            continue
        for tf in TIMEFRAMES:
            t0 = time.time()
            context_tf = CONTEXT_TF.get(tf, "4h")
            context_df = build_bars_for_tf(df_5m, context_tf)
            trading_bars = build_bars_for_tf(df_5m, tf)

            sigs = collect_signals(trading_bars, context_df, sym, tf)
            all_signals.extend(sigs)

            highs = trading_bars["high"].values.astype(np.float64)
            lows = trading_bars["low"].values.astype(np.float64)
            closes = trading_bars["close"].values.astype(np.float64)
            bar_data[(sym, tf)] = (highs, lows, closes, len(trading_bars))

            elapsed = time.time() - t0
            print(f"  {tf} {sym}: {len(sigs)} signals  ({elapsed:.1f}s)",
                  flush=True)

    print(f"\n  Total baseline signals: {len(all_signals)}", flush=True)

    # ADX distribution
    adx_vals = [s.adx for s in all_signals if s.adx > 0]
    if adx_vals:
        print(f"  ADX distribution: min={min(adx_vals):.1f}  "
              f"median={sorted(adx_vals)[len(adx_vals)//2]:.1f}  "
              f"mean={sum(adx_vals)/len(adx_vals):.1f}  "
              f"max={max(adx_vals):.1f}", flush=True)

    bb_vals = [s.bb_width for s in all_signals if s.bb_width > 0]
    if bb_vals:
        print(f"  BB width distribution: min={min(bb_vals):.4f}  "
              f"median={sorted(bb_vals)[len(bb_vals)//2]:.4f}  "
              f"mean={sum(bb_vals)/len(bb_vals):.4f}  "
              f"max={max(bb_vals):.4f}", flush=True)

    # ── Test 1: ADX filter only ──
    print("\n" + "=" * 80, flush=True)
    print("TEST 1: ADX FILTER (min ADX to take a trade)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'ADX >':>7s}  {'Trades':>7s}  {'Kept%':>6s}  {'WR':>6s}  "
          f"{'PnL':>12s}  {'Return':>8s}  {'TP':>5s}  {'SL':>6s}  "
          f"{'Trail':>5s}  {'PF':>6s}", flush=True)
    print("-" * 80, flush=True)

    adx_results = []
    for threshold in adx_thresholds:
        # Group signals by (symbol, tf) and replay each independently
        total = {"trades": 0, "wins": 0, "pnl": 0.0,
                 "tp": 0, "sl": 0, "trail": 0}

        for (sym, tf), (highs, lows, closes, n) in bar_data.items():
            filtered = [s for s in all_signals
                        if s.symbol == sym and s.tf == tf
                        and s.adx >= threshold]
            r = replay(filtered, highs, lows, closes, n)
            total["trades"] += r["trades"]
            total["wins"] += r["wins"]
            total["pnl"] += r["pnl"]
            total["tp"] += r["tp"]
            total["sl"] += r["sl"]
            total["trail"] += r["trail"]

        kept = total["trades"]
        baseline_trades = sum(1 for _ in all_signals)
        wr = total["wins"] / total["trades"] * 100 if total["trades"] else 0
        ret = total["pnl"] / EQUITY * 100
        win_pnl = total["pnl"] + abs(total["sl"]) if total["pnl"] > 0 else 0
        pf = (total["tp"] + total["trail"]) / total["sl"] if total["sl"] else 0
        # Proper PF from actual PnL
        # We need wins/losses dollar amounts
        print(f"{threshold:>7d}  {total['trades']:>7d}  "
              f"{total['trades']/len(all_signals)*100 if all_signals else 0:>5.1f}%  "
              f"{wr:>5.1f}%  ${total['pnl']:>+10,.2f}  {ret:>+7.1f}%  "
              f"{total['tp']:>5d}  {total['sl']:>6d}  "
              f"{total['trail']:>5d}  "
              f"{(total['tp']+total['trail'])/total['sl'] if total['sl'] else 0:>5.2f}",
              flush=True)
        adx_results.append({"threshold": threshold, **total, "wr": wr})

    # ── Test 2: BB width filter only ──
    print("\n" + "=" * 80, flush=True)
    print("TEST 2: BB WIDTH FILTER (min bb_width to take a trade)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'BBW >':>7s}  {'Trades':>7s}  {'Kept%':>6s}  {'WR':>6s}  "
          f"{'PnL':>12s}  {'Return':>8s}  {'TP':>5s}  {'SL':>6s}  "
          f"{'Trail':>5s}", flush=True)
    print("-" * 80, flush=True)

    for threshold in bb_width_thresholds:
        total = {"trades": 0, "wins": 0, "pnl": 0.0,
                 "tp": 0, "sl": 0, "trail": 0}

        for (sym, tf), (highs, lows, closes, n) in bar_data.items():
            filtered = [s for s in all_signals
                        if s.symbol == sym and s.tf == tf
                        and s.bb_width >= threshold]
            r = replay(filtered, highs, lows, closes, n)
            total["trades"] += r["trades"]
            total["wins"] += r["wins"]
            total["pnl"] += r["pnl"]
            total["tp"] += r["tp"]
            total["sl"] += r["sl"]
            total["trail"] += r["trail"]

        wr = total["wins"] / total["trades"] * 100 if total["trades"] else 0
        ret = total["pnl"] / EQUITY * 100
        print(f"{threshold:>7.2f}  {total['trades']:>7d}  "
              f"{total['trades']/len(all_signals)*100 if all_signals else 0:>5.1f}%  "
              f"{wr:>5.1f}%  ${total['pnl']:>+10,.2f}  {ret:>+7.1f}%  "
              f"{total['tp']:>5d}  {total['sl']:>6d}  "
              f"{total['trail']:>5d}", flush=True)

    # ── Test 3: BB squeeze filter (skip when BB squeeze active) ──
    print("\n" + "=" * 80, flush=True)
    print("TEST 3: BB SQUEEZE FILTER (skip trades when squeeze is active)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Filter':>12s}  {'Trades':>7s}  {'Kept%':>6s}  {'WR':>6s}  "
          f"{'PnL':>12s}  {'Return':>8s}", flush=True)
    print("-" * 60, flush=True)

    for label, use_filter in [("No filter", False), ("Skip squeeze", True)]:
        total = {"trades": 0, "wins": 0, "pnl": 0.0}

        for (sym, tf), (highs, lows, closes, n) in bar_data.items():
            if use_filter:
                filtered = [s for s in all_signals
                            if s.symbol == sym and s.tf == tf
                            and not s.bb_squeeze]
            else:
                filtered = [s for s in all_signals
                            if s.symbol == sym and s.tf == tf]
            r = replay(filtered, highs, lows, closes, n)
            total["trades"] += r["trades"]
            total["wins"] += r["wins"]
            total["pnl"] += r["pnl"]

        wr = total["wins"] / total["trades"] * 100 if total["trades"] else 0
        ret = total["pnl"] / EQUITY * 100
        print(f"{label:>12s}  {total['trades']:>7d}  "
              f"{total['trades']/len(all_signals)*100 if all_signals else 0:>5.1f}%  "
              f"{wr:>5.1f}%  ${total['pnl']:>+10,.2f}  {ret:>+7.1f}%",
              flush=True)

    # ── Test 4: Best ADX + best BB combos ──
    print("\n" + "=" * 80, flush=True)
    print("TEST 4: COMBINED FILTERS (ADX + BB width)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'ADX >':>7s}  {'BBW >':>7s}  {'Trades':>7s}  {'Kept%':>6s}  "
          f"{'WR':>6s}  {'PnL':>12s}  {'Return':>8s}", flush=True)
    print("-" * 65, flush=True)

    combo_adx = [0, 20, 22, 25]
    combo_bb = [0, 0.02, 0.03, 0.04]

    best_pnl = -999999
    best_combo = ""

    for adx_t in combo_adx:
        for bb_t in combo_bb:
            total = {"trades": 0, "wins": 0, "pnl": 0.0}

            for (sym, tf), (highs, lows, closes, n) in bar_data.items():
                filtered = [s for s in all_signals
                            if s.symbol == sym and s.tf == tf
                            and s.adx >= adx_t and s.bb_width >= bb_t]
                r = replay(filtered, highs, lows, closes, n)
                total["trades"] += r["trades"]
                total["wins"] += r["wins"]
                total["pnl"] += r["pnl"]

            wr = total["wins"] / total["trades"] * 100 if total["trades"] else 0
            ret = total["pnl"] / EQUITY * 100
            marker = ""
            if total["pnl"] > best_pnl:
                best_pnl = total["pnl"]
                best_combo = f"ADX>{adx_t} + BBW>{bb_t}"
                marker = " <-- best"

            print(f"{adx_t:>7d}  {bb_t:>7.2f}  {total['trades']:>7d}  "
                  f"{total['trades']/len(all_signals)*100 if all_signals else 0:>5.1f}%  "
                  f"{wr:>5.1f}%  ${total['pnl']:>+10,.2f}  "
                  f"{ret:>+7.1f}%{marker}", flush=True)

    print(f"\n  Best combo: {best_combo}  (PnL: ${best_pnl:+,.2f})", flush=True)
    print("\n" + "=" * 80, flush=True)


if __name__ == "__main__":
    main()
