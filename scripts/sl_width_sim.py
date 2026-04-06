"""
SL-width simulation: test wider stop losses with smaller positions.

For each SL multiplier, we regenerate signals (wider SL changes which
trades trigger), then run the portfolio sim on shared equity.

Two modes tested:
  A) Fixed R:R (5.4:1) -- TP scales with SL so reward/risk ratio stays same
  B) Fixed TP mult (6.5) -- only SL widens, R:R decreases but win rate rises

Usage:
    python scripts/sl_width_sim.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.strategy_sniper import SniperStrategy

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
RISK_PCT = 0.01
MAX_CONCURRENT = 4

SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT",
    "1000PEPEUSDT", "DOGEUSDT", "OPUSDT", "SUIUSDT",
]
TIMEFRAMES = ["15m", "4h"]
FROM = "2023-01-01"
TO = "2026-04-03"

RR_RATIO = 5.4
FIXED_TP_MULT = 6.5

SL_MULTS_TO_TEST = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]


@dataclass
class PrecomputedTrade:
    symbol: str
    tf: str
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    sl_distance: float
    tp_distance: float
    raw_pct: float
    exit_reason: str
    bars_held: int


def collect_and_resolve(trading_bars, context_df, symbol, tf, sl_mult, tp_mult):
    sniper = SniperStrategy(sl_atr_mult=sl_mult, tp_atr_mult=tp_mult)
    trades = []

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

    highs = trading_bars["high"].values.astype(np.float64)
    lows = trading_bars["low"].values.astype(np.float64)
    closes = trading_bars["close"].values.astype(np.float64)
    timestamps = trading_bars["timestamp"].values
    n = len(trading_bars)
    last_exit_idx = -COOLDOWN_BARS - 1

    for i in range(n):
        if i <= last_exit_idx + COOLDOWN_BARS:
            continue

        bar = trading_bars.iloc[i]
        ctx = get_ctx(pd.Timestamp(bar.get("timestamp")))
        sig = sniper.generate_signal(
            symbol=symbol, indicators_5m=bar,
            indicators_15m=ctx, funding_rate=0.0,
            liq_volume_1h=0.0,
        )
        if sig is None or sig.direction == "flat":
            continue

        entry = sig.entry_price
        sl_dist = abs(entry - sig.stop_loss)
        tp_dist = abs(entry - sig.take_profit)
        if sl_dist <= 0 or entry <= 0:
            continue

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
        exit_reason = None
        exit_idx = n - 1

        for j in range(i + 1, n):
            bh, bl = highs[j], lows[j]
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
                exit_idx = j
                break
            elif tp_hit:
                exit_price = tp
                exit_reason = "tp"
                exit_idx = j
                break

        if exit_price is None:
            exit_price = closes[-1]
            exit_reason = "timeout"
            exit_idx = n - 1

        if sig.direction == "long":
            raw_pct = (exit_price - entry) / entry
        else:
            raw_pct = (entry - exit_price) / entry

        last_exit_idx = exit_idx
        trades.append(PrecomputedTrade(
            symbol=symbol, tf=tf, direction=sig.direction,
            entry_time=pd.Timestamp(timestamps[i]),
            exit_time=pd.Timestamp(timestamps[exit_idx]),
            entry_price=entry, exit_price=exit_price,
            sl_distance=sl_dist, tp_distance=tp_dist,
            raw_pct=raw_pct, exit_reason=exit_reason,
            bars_held=exit_idx - i,
        ))

    return trades


def run_portfolio(all_trades: list[PrecomputedTrade]) -> dict:
    equity = START_EQUITY
    peak = equity
    max_dd_pct = 0.0
    max_dd_usd = 0.0
    min_equity = equity

    events = []
    for idx, t in enumerate(all_trades):
        events.append(("entry", t.entry_time, idx))
        events.append(("exit", t.exit_time, idx))
    events.sort(key=lambda e: (e[1], 0 if e[0] == "exit" else 1))

    open_positions: dict[int, dict] = {}
    closed_trades = 0
    wins = 0
    total_pnl = 0.0
    total_fees = 0.0
    open_symbols: set[str] = set()
    max_concurrent_seen = 0
    skipped = 0
    tp_count = sl_count = trail_count = 0

    for event_type, event_time, trade_idx in events:
        trade = all_trades[trade_idx]

        if event_type == "exit" and trade_idx in open_positions:
            pos = open_positions.pop(trade_idx)
            notional = pos["notional"]
            entry_fee = pos["entry_fee"]
            exit_fee = notional * TAKER_FEE
            pnl = notional * trade.raw_pct - entry_fee - exit_fee
            equity += pnl
            total_pnl += pnl
            total_fees += entry_fee + exit_fee
            closed_trades += 1
            if pnl > 0:
                wins += 1
            if trade.exit_reason == "tp":
                tp_count += 1
            elif trade.exit_reason == "trail":
                trail_count += 1
            elif trade.exit_reason == "sl":
                sl_count += 1

            open_symbols.discard(f"{trade.symbol}_{trade.tf}")
            if equity > peak:
                peak = equity
            dd_pct = (peak - equity) / peak if peak > 0 else 0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_usd = peak - equity
            if equity < min_equity:
                min_equity = equity

        elif event_type == "entry":
            sym_key = f"{trade.symbol}_{trade.tf}"
            if (len(open_positions) >= MAX_CONCURRENT or
                    sym_key in open_symbols or equity <= 0):
                skipped += 1
                continue

            risk_amount = equity * RISK_PCT
            qty = risk_amount / (trade.sl_distance +
                                 trade.entry_price * ROUND_TRIP_COST)
            notional = qty * trade.entry_price
            max_notional = equity * LEVERAGE * NOTIONAL_CAP_FRAC
            if notional > max_notional:
                notional = max_notional

            open_positions[trade_idx] = {
                "notional": notional,
                "entry_fee": notional * MAKER_FEE,
            }
            open_symbols.add(sym_key)
            if len(open_positions) > max_concurrent_seen:
                max_concurrent_seen = len(open_positions)

    for trade_idx, pos in list(open_positions.items()):
        trade = all_trades[trade_idx]
        notional = pos["notional"]
        exit_fee = notional * TAKER_FEE
        pnl = notional * trade.raw_pct - pos["entry_fee"] - exit_fee
        equity += pnl
        total_pnl += pnl
        closed_trades += 1
        if pnl > 0:
            wins += 1

    if equity > peak:
        peak = equity
    dd_pct = (peak - equity) / peak if peak > 0 else 0
    if dd_pct > max_dd_pct:
        max_dd_pct = dd_pct
    if equity < min_equity:
        min_equity = equity

    wr = wins / closed_trades * 100 if closed_trades else 0
    return {
        "trades": closed_trades, "wins": wins, "wr": wr,
        "pnl": total_pnl, "final_eq": equity,
        "total_fees": total_fees,
        "max_dd_pct": max_dd_pct, "max_dd_usd": max_dd_usd,
        "min_equity": min_equity,
        "max_concurrent": max_concurrent_seen,
        "skipped": skipped,
        "tp": tp_count, "sl": sl_count, "trail": trail_count,
    }


def load_candles(symbol, cache_dir):
    cache_path = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return pd.DataFrame()


def build_all_bars(cache_dir):
    """Pre-build trading/context bars for each (symbol, tf)."""
    bar_sets = {}
    for sym in SYMBOLS:
        df_5m = load_candles(sym, cache_dir)
        if df_5m.empty:
            continue
        for tf in TIMEFRAMES:
            context_tf = CONTEXT_TF.get(tf, "4h")
            context_df = build_bars_for_tf(df_5m, context_tf)
            trading_bars = build_bars_for_tf(df_5m, tf)
            bar_sets[(sym, tf)] = (trading_bars, context_df)
    return bar_sets


def run_scenario(bar_sets, sl_mult, tp_mult, label):
    all_trades = []
    for (sym, tf), (trading_bars, context_df) in bar_sets.items():
        trades = collect_and_resolve(
            trading_bars, context_df, sym, tf, sl_mult, tp_mult)
        all_trades.extend(trades)
    all_trades.sort(key=lambda t: t.entry_time)
    result = run_portfolio(all_trades)
    result["label"] = label
    result["sl_mult"] = sl_mult
    result["tp_mult"] = tp_mult
    result["rr"] = tp_mult / sl_mult
    result["total_signals"] = len(all_trades)
    return result


def main():
    cache_dir = project_root / "data_cache"

    print("=" * 90, flush=True)
    print("SL WIDTH SIMULATION: wider SL = smaller position = more breathing room", flush=True)
    print(f"Equity: ${START_EQUITY:,.0f}  |  Risk: {RISK_PCT:.0%}  |  "
          f"Leverage: {LEVERAGE}x  |  Max concurrent: {MAX_CONCURRENT}", flush=True)
    print("=" * 90, flush=True)

    print("\n-- Building bars (one-time) --", flush=True)
    t0 = time.time()
    bar_sets = build_all_bars(cache_dir)
    print(f"   Done in {time.time() - t0:.1f}s  "
          f"({len(bar_sets)} symbol/tf combos)\n", flush=True)

    # ── Mode A: Fixed R:R (5.4:1) ──
    print("=" * 90, flush=True)
    print(f"MODE A: Fixed R:R = {RR_RATIO}:1  (TP scales with SL)", flush=True)
    print("=" * 90, flush=True)

    results_a = []
    for sl_m in SL_MULTS_TO_TEST:
        tp_m = sl_m * RR_RATIO
        label = f"SL={sl_m:.1f}x / TP={tp_m:.1f}x"
        print(f"  Running {label} ...", end=" ", flush=True)
        t0 = time.time()
        r = run_scenario(bar_sets, sl_m, tp_m, label)
        results_a.append(r)
        print(f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{'SL mult':>8s}  {'TP mult':>8s}  {'R:R':>5s}  "
          f"{'Signals':>8s}  {'Trades':>7s}  {'WR':>6s}  "
          f"{'Final Eq':>10s}  {'PnL':>12s}  {'Return':>8s}  "
          f"{'Max DD':>7s}  {'Min Eq':>9s}", flush=True)
    print("-" * 110, flush=True)
    for r in results_a:
        ret = r["pnl"] / START_EQUITY * 100
        print(f"{r['sl_mult']:>7.1f}x  {r['tp_mult']:>7.1f}x  "
              f"{r['rr']:>4.1f}:1  "
              f"{r['total_signals']:>8d}  {r['trades']:>7d}  "
              f"{r['wr']:>5.1f}%  "
              f"${r['final_eq']:>8,.0f}  ${r['pnl']:>+10,.0f}  "
              f"{ret:>+7.1f}%  {r['max_dd_pct']:>6.1%}  "
              f"${r['min_equity']:>7,.0f}", flush=True)

    # ── Mode B: Fixed TP (6.5x ATR), only SL widens ──
    print(f"\n\n{'=' * 90}", flush=True)
    print(f"MODE B: Fixed TP = {FIXED_TP_MULT}x ATR  (only SL widens, R:R decreases)",
          flush=True)
    print("=" * 90, flush=True)

    results_b = []
    for sl_m in SL_MULTS_TO_TEST:
        tp_m = FIXED_TP_MULT
        label = f"SL={sl_m:.1f}x / TP={tp_m:.1f}x"
        print(f"  Running {label} ...", end=" ", flush=True)
        t0 = time.time()
        r = run_scenario(bar_sets, sl_m, tp_m, label)
        results_b.append(r)
        print(f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{'SL mult':>8s}  {'TP mult':>8s}  {'R:R':>5s}  "
          f"{'Signals':>8s}  {'Trades':>7s}  {'WR':>6s}  "
          f"{'Final Eq':>10s}  {'PnL':>12s}  {'Return':>8s}  "
          f"{'Max DD':>7s}  {'Min Eq':>9s}", flush=True)
    print("-" * 110, flush=True)
    for r in results_b:
        ret = r["pnl"] / START_EQUITY * 100
        print(f"{r['sl_mult']:>7.1f}x  {r['tp_mult']:>7.1f}x  "
              f"{r['rr']:>4.1f}:1  "
              f"{r['total_signals']:>8d}  {r['trades']:>7d}  "
              f"{r['wr']:>5.1f}%  "
              f"${r['final_eq']:>8,.0f}  ${r['pnl']:>+10,.0f}  "
              f"{ret:>+7.1f}%  {r['max_dd_pct']:>6.1%}  "
              f"${r['min_equity']:>7,.0f}", flush=True)

    # Best from each mode
    best_a = max(results_a, key=lambda r: r["final_eq"])
    best_b = max(results_b, key=lambda r: r["final_eq"])
    print(f"\n{'=' * 90}", flush=True)
    print("BEST RESULTS", flush=True)
    print(f"  Mode A (fixed R:R):  {best_a['label']}  ->  "
          f"${best_a['final_eq']:,.0f}  (WR {best_a['wr']:.1f}%  "
          f"DD {best_a['max_dd_pct']:.1%})", flush=True)
    print(f"  Mode B (fixed TP):   {best_b['label']}  ->  "
          f"${best_b['final_eq']:,.0f}  (WR {best_b['wr']:.1f}%  "
          f"DD {best_b['max_dd_pct']:.1%})", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
