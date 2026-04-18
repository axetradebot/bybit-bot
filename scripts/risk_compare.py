"""
Compare 1% vs 2% risk per trade on the same signal set.

Tracks equity curve, max drawdown, and near-wipe events.

Usage:
    python scripts/risk_compare.py
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
LEVERAGE = 20
TRAIL_ACTIVATE = 0.10
TRAIL_OFFSET = 0.03
START_EQUITY = 3000.0

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
    symbol: str
    tf: str
    timestamp: str


def collect_signals(trading_bars, context_df, symbol, tf):
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
            signals.append(Signal(
                bar_idx=i, direction=sig.direction,
                entry_price=sig.entry_price,
                sl_distance=sl_dist, tp_distance=tp_dist,
                symbol=symbol, tf=tf,
                timestamp=str(bar.get("timestamp")),
            ))
    return signals


def replay_with_equity_curve(
    signals: list[Signal], highs, lows, closes, n_bars,
    risk_pct: float,
) -> dict:
    equity = START_EQUITY
    peak = equity
    max_dd_pct = 0.0
    max_dd_usd = 0.0
    min_equity = equity
    trades = 0
    wins = 0
    total_pnl = 0.0
    last_exit_idx = -COOLDOWN_BARS - 1
    equity_snapshots = []
    consecutive_losses = 0
    max_consecutive_losses = 0

    for sig in signals:
        if sig.bar_idx <= last_exit_idx + COOLDOWN_BARS:
            continue

        entry = sig.entry_price
        sl_dist = sig.sl_distance
        tp_dist = sig.tp_distance

        if sl_dist <= 0 or entry <= 0 or equity <= 0:
            continue

        risk_amount = equity * risk_pct
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

        trail_active = False
        hwm = 0.0
        current_sl = sl
        exit_price = None

        for j in range(sig.bar_idx + 1, n_bars):
            bh = highs[j]
            bl = lows[j]

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
                last_exit_idx = j
                break
            elif tp_hit:
                exit_price = tp
                last_exit_idx = j
                break

        if exit_price is None:
            exit_price = closes[-1]
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
            consecutive_losses = 0
        else:
            consecutive_losses += 1
            if consecutive_losses > max_consecutive_losses:
                max_consecutive_losses = consecutive_losses

        if equity > peak:
            peak = equity
        dd_pct = (peak - equity) / peak if peak > 0 else 0
        dd_usd = peak - equity
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_usd = dd_usd
        if equity < min_equity:
            min_equity = equity

        equity_snapshots.append({
            "trade_num": trades,
            "equity": equity,
            "pnl": pnl,
            "dd_pct": dd_pct,
            "timestamp": sig.timestamp,
        })

    wr = wins / trades * 100 if trades else 0
    return {
        "trades": trades, "wins": wins, "wr": wr,
        "pnl": total_pnl, "final_eq": equity,
        "max_dd_pct": max_dd_pct, "max_dd_usd": max_dd_usd,
        "min_equity": min_equity,
        "max_consecutive_losses": max_consecutive_losses,
        "snapshots": equity_snapshots,
    }


def load_candles(symbol, cache_dir):
    cache_path = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return pd.DataFrame()


def main():
    cache_dir = project_root / "data_cache"

    print("=" * 75, flush=True)
    print("RISK COMPARISON: 1% vs 2% per trade", flush=True)
    print(f"Starting equity: ${START_EQUITY:,.0f}  |  Leverage: {LEVERAGE}x  "
          f"|  Cap: {NOTIONAL_CAP_FRAC:.0%}", flush=True)
    print("=" * 75, flush=True)

    # Collect all signals
    print("\n-- Collecting signals --", flush=True)
    all_signals: dict[tuple[str, str], list[Signal]] = {}
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
            all_signals[(sym, tf)] = sigs

            highs = trading_bars["high"].values.astype(np.float64)
            lows = trading_bars["low"].values.astype(np.float64)
            closes = trading_bars["close"].values.astype(np.float64)
            bar_data[(sym, tf)] = (highs, lows, closes, len(trading_bars))

            elapsed = time.time() - t0
            print(f"  {tf} {sym}: {len(sigs)} signals  ({elapsed:.1f}s)",
                  flush=True)

    risk_levels = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]

    print("\n-- Replaying at different risk levels --", flush=True)

    for risk_pct in risk_levels:
        agg = {
            "trades": 0, "wins": 0, "pnl": 0.0,
            "max_dd_pct": 0.0, "max_dd_usd": 0.0,
            "min_equity": START_EQUITY,
            "max_consec": 0,
            "wipe_events": 0,
        }
        per_pair: list[dict] = []

        for (sym, tf), (highs, lows, closes, n) in bar_data.items():
            sigs = all_signals.get((sym, tf), [])
            r = replay_with_equity_curve(sigs, highs, lows, closes, n, risk_pct)
            agg["trades"] += r["trades"]
            agg["wins"] += r["wins"]
            agg["pnl"] += r["pnl"]
            if r["max_dd_pct"] > agg["max_dd_pct"]:
                agg["max_dd_pct"] = r["max_dd_pct"]
                agg["max_dd_usd"] = r["max_dd_usd"]
            if r["min_equity"] < agg["min_equity"]:
                agg["min_equity"] = r["min_equity"]
            if r["max_consecutive_losses"] > agg["max_consec"]:
                agg["max_consec"] = r["max_consecutive_losses"]
            if r["min_equity"] < START_EQUITY * 0.10:
                agg["wipe_events"] += 1

            per_pair.append({
                "sym": sym, "tf": tf,
                "pnl": r["pnl"], "final_eq": r["final_eq"],
                "max_dd": r["max_dd_pct"],
                "min_eq": r["min_equity"],
            })

        wr = agg["wins"] / agg["trades"] * 100 if agg["trades"] else 0
        final = START_EQUITY + agg["pnl"]
        ret = agg["pnl"] / START_EQUITY * 100

        print(f"\n  Risk: {risk_pct:.1%} per trade  (done)", flush=True)

        # Count pairs where equity dropped below danger thresholds
        below_50 = sum(1 for p in per_pair if p["min_eq"] < START_EQUITY * 0.50)
        below_25 = sum(1 for p in per_pair if p["min_eq"] < START_EQUITY * 0.25)
        below_10 = sum(1 for p in per_pair if p["min_eq"] < START_EQUITY * 0.10)

    # Summary table
    print("\n\n" + "=" * 75, flush=True)
    print("RESULTS COMPARISON", flush=True)
    print("=" * 75, flush=True)
    print(f"{'Risk%':>6s}  {'Trades':>7s}  {'WR':>6s}  {'Total PnL':>14s}  "
          f"{'Return':>8s}  {'Max DD':>8s}  {'Min Eq':>10s}  "
          f"{'MaxLoss':>8s}  {'Wipes':>5s}", flush=True)
    print("-" * 75, flush=True)

    for risk_pct in risk_levels:
        agg = {
            "trades": 0, "wins": 0, "pnl": 0.0,
            "max_dd_pct": 0.0, "max_dd_usd": 0.0,
            "min_equity": START_EQUITY,
            "max_consec": 0,
        }
        wipe_count = 0

        for (sym, tf), (highs, lows, closes, n) in bar_data.items():
            sigs = all_signals.get((sym, tf), [])
            r = replay_with_equity_curve(sigs, highs, lows, closes, n, risk_pct)
            agg["trades"] += r["trades"]
            agg["wins"] += r["wins"]
            agg["pnl"] += r["pnl"]
            if r["max_dd_pct"] > agg["max_dd_pct"]:
                agg["max_dd_pct"] = r["max_dd_pct"]
                agg["max_dd_usd"] = r["max_dd_usd"]
            if r["min_equity"] < agg["min_equity"]:
                agg["min_equity"] = r["min_equity"]
            if r["max_consecutive_losses"] > agg["max_consec"]:
                agg["max_consec"] = r["max_consecutive_losses"]
            if r["min_equity"] < START_EQUITY * 0.10:
                wipe_count += 1

        wr = agg["wins"] / agg["trades"] * 100 if agg["trades"] else 0
        ret = agg["pnl"] / START_EQUITY * 100
        marker = " <--" if risk_pct == 0.02 else ""

        print(f"{risk_pct:>5.1%}  {agg['trades']:>7d}  {wr:>5.1f}%  "
              f"${agg['pnl']:>+12,.2f}  {ret:>+7.1f}%  "
              f"{agg['max_dd_pct']:>7.1%}  ${agg['min_equity']:>8,.2f}  "
              f"{agg['max_consec']:>7d}  {wipe_count:>5d}{marker}",
              flush=True)

    # Detailed per-pair danger analysis at 2% and 1%
    print("\n\n" + "=" * 75, flush=True)
    print("ACCOUNT DANGER ANALYSIS (per pair, independent equity)", flush=True)
    print("=" * 75, flush=True)

    for risk_pct in [0.01, 0.02]:
        print(f"\n  At {risk_pct:.0%} risk:", flush=True)
        print(f"  {'Pair':<20s}  {'Final Eq':>10s}  {'Min Eq':>10s}  "
              f"{'Max DD':>8s}  {'MaxLoss':>7s}  {'Status':>10s}", flush=True)
        print("  " + "-" * 70, flush=True)

        for (sym, tf), (highs, lows, closes, n) in sorted(bar_data.items()):
            sigs = all_signals.get((sym, tf), [])
            r = replay_with_equity_curve(sigs, highs, lows, closes, n, risk_pct)

            if r["min_equity"] < START_EQUITY * 0.10:
                status = "!! WIPED"
            elif r["min_equity"] < START_EQUITY * 0.25:
                status = "! DANGER"
            elif r["min_equity"] < START_EQUITY * 0.50:
                status = "WARNING"
            else:
                status = "OK"

            print(f"  {tf:>3s} {sym:<16s}  ${r['final_eq']:>8,.2f}  "
                  f"${r['min_equity']:>8,.2f}  {r['max_dd_pct']:>7.1%}  "
                  f"{r['max_consecutive_losses']:>7d}  {status:>10s}",
                  flush=True)

    print("\n" + "=" * 75, flush=True)


if __name__ == "__main__":
    main()
