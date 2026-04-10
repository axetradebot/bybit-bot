"""
MULTI-TIMEFRAME POSITION SIZING SIMULATION

Tests whether using different risk% per timeframe improves results:
  - Flat:  2.5% for all TFs (current)
  - Split: 1.5% for 15m, 3.5% for 4h (higher conviction = bigger size)
  - Aggressive split: 1.0% for 15m, 4.0% for 4h
  - Conservative split: 2.0% for 15m, 3.0% for 4h

Uses the best bear-survival filter (ADX>=25 + DI + BB mid + vol>=2.0).

Run:
    python scripts/tf_sizing_sim.py
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

from src.backtest.live_aligned_portfolio import (
    LivePortfolioTrade,
    collect_trades_live_aligned,
    run_portfolio_live_aligned,
    MAKER_FEE, TAKER_FEE, SLIPPAGE_BPS, LEVERAGE, NOTIONAL_CAP_FRAC,
    ROUND_TRIP_COST_SIZING, FUNDING_RATE_8H_FLAT,
    _funding_usd,
)
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.base import _sf

START_EQUITY = 3_000.0
MAX_CONCURRENT = 4

SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT", "1000PEPEUSDT",
    "DOGEUSDT", "OPUSDT", "BTCUSDT", "XRPUSDT",
]
TIMEFRAMES = ["15m", "4h"]
FROM = "2023-01-01"
TO = "2026-04-03"

WINDOWS = {
    "FULL":      (FROM, TO),
    "BEAR-2023": ("2023-01-01", "2023-12-31"),
    "BEAR-LATE": ("2025-11-01", "2026-04-03"),
}

_CHOP_ADX_FLOOR = 25
_CHOP_VOL_FLOOR = 2.0

SIZING_SCENARIOS = [
    ("Flat 2.5%",            {"15m": 0.025, "4h": 0.025}),
    ("Split 1.5%/3.5%",      {"15m": 0.015, "4h": 0.035}),
    ("Split 1.0%/4.0%",      {"15m": 0.010, "4h": 0.040}),
    ("Split 2.0%/3.0%",      {"15m": 0.020, "4h": 0.030}),
    ("Aggressive 0.5%/5.0%", {"15m": 0.005, "4h": 0.050}),
    ("4h only (15m=0%)",     {"15m": 0.000, "4h": 0.035}),
    ("15m heavy 3.0%/2.0%",  {"15m": 0.030, "4h": 0.020}),
]


def _snap(bar, ctx):
    return {
        "adx_14": _sf(bar.get("adx_14")),
        "plus_di": _sf(bar.get("plus_di")),
        "minus_di": _sf(bar.get("minus_di")),
        "bb_mid": _sf(bar.get("bb_mid")),
        "volume_ratio": _sf(bar.get("volume_ratio")),
        "close": _sf(bar.get("close")),
    }


def _passes_filter(snap, direction):
    adx = snap["adx_14"]
    if adx > 0 and adx < _CHOP_ADX_FLOOR:
        return False
    p, m = snap["plus_di"], snap["minus_di"]
    if p > 0 or m > 0:
        if direction == "long" and p <= m:
            return False
        if direction == "short" and m <= p:
            return False
    bb_mid, close = snap["bb_mid"], snap["close"]
    if bb_mid > 0 and close > 0:
        if direction == "long" and close <= bb_mid:
            return False
        if direction == "short" and close >= bb_mid:
            return False
    vr = snap["volume_ratio"]
    if vr > 0 and vr < _CHOP_VOL_FLOOR:
        return False
    return True


def run_portfolio_tf_sizing(
    trades: list[LivePortfolioTrade],
    start_equity: float,
    risk_by_tf: dict[str, float],
    max_concurrent: int = 4,
) -> dict:
    """Like run_portfolio_live_aligned but with per-TF risk sizing."""
    equity = start_equity
    peak = equity
    max_dd_pct = max_dd_usd = 0.0
    min_equity = equity

    events = []
    for idx, t in enumerate(trades):
        events.append(("entry", t.entry_time, idx))
        events.append(("exit", t.exit_time, idx))
    events.sort(key=lambda e: (e[1], 0 if e[0] == "exit" else 1))

    open_positions: dict[int, dict] = {}
    closed = wins = 0
    total_pnl = total_fees = total_funding = 0.0
    open_symbols: set[str] = set()
    skipped = 0
    tp_count = sl_count = trail_count = timeout_count = 0
    tf_trades = {"15m": 0, "4h": 0}

    def exit_fee_rate(reason):
        return MAKER_FEE if reason == "tp" else TAKER_FEE

    for event_type, event_time, trade_idx in events:
        trade = trades[trade_idx]

        if event_type == "exit" and trade_idx in open_positions:
            pos = open_positions.pop(trade_idx)
            notional = pos["notional"]
            entry_fee = pos["entry_fee"]
            efr = exit_fee_rate(trade.exit_reason)
            exit_fee = notional * efr
            fund = _funding_usd(notional, trade.bars_held, trade.tf, trade.funding_rate_8h)
            pnl = notional * trade.raw_pct - entry_fee - exit_fee - fund
            equity += pnl
            total_pnl += pnl
            total_fees += entry_fee + exit_fee
            total_funding += fund
            closed += 1
            if pnl > 0:
                wins += 1
            if trade.exit_reason == "tp": tp_count += 1
            elif trade.exit_reason == "trail": trail_count += 1
            elif trade.exit_reason == "sl": sl_count += 1
            elif trade.exit_reason == "timeout": timeout_count += 1
            open_symbols.discard(f"{trade.symbol}_{trade.tf}")
            if equity > peak: peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd_pct:
                max_dd_pct = dd
                max_dd_usd = peak - equity
            if equity < min_equity: min_equity = equity

        elif event_type == "entry":
            sym_key = f"{trade.symbol}_{trade.tf}"
            rp = risk_by_tf.get(trade.tf, 0.025)
            if rp <= 0:
                skipped += 1
                continue
            if len(open_positions) >= max_concurrent or sym_key in open_symbols or equity <= 0:
                skipped += 1
                continue
            sl_dist = trade.sl_distance
            if sl_dist <= 0:
                skipped += 1
                continue
            risk_amount = equity * rp
            qty = risk_amount / (sl_dist + trade.entry_price * ROUND_TRIP_COST_SIZING)
            notional = qty * trade.entry_price
            max_notional = equity * LEVERAGE * NOTIONAL_CAP_FRAC
            if notional > max_notional:
                notional = max_notional
            entry_fee = notional * TAKER_FEE
            open_positions[trade_idx] = {"notional": notional, "entry_fee": entry_fee}
            open_symbols.add(sym_key)
            tf_trades[trade.tf] = tf_trades.get(trade.tf, 0) + 1

    for trade_idx, pos in list(open_positions.items()):
        trade = trades[trade_idx]
        notional = pos["notional"]
        efr = exit_fee_rate(trade.exit_reason)
        exit_fee = notional * efr
        fund = _funding_usd(notional, trade.bars_held, trade.tf, trade.funding_rate_8h)
        pnl = notional * trade.raw_pct - pos["entry_fee"] - exit_fee - fund
        equity += pnl
        total_pnl += pnl
        total_fees += pos["entry_fee"] + exit_fee
        total_funding += fund
        closed += 1
        if pnl > 0: wins += 1
        if equity > peak: peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        if dd > max_dd_pct:
            max_dd_pct = dd
            max_dd_usd = peak - equity
        if equity < min_equity: min_equity = equity

    wr = wins / closed * 100 if closed else 0
    return {
        "trades": closed, "wins": wins, "wr": wr,
        "pnl": total_pnl, "final_eq": equity,
        "total_fees": total_fees, "total_funding": total_funding,
        "max_dd_pct": max_dd_pct, "max_dd_usd": max_dd_usd,
        "min_equity": min_equity, "skipped": skipped,
        "tp": tp_count, "sl": sl_count, "trail": trail_count, "timeout": timeout_count,
        "tf_trades": tf_trades,
    }


def load_candles(symbol, cache_dir):
    p = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _filter_by_window(trades, ws, we):
    wst = pd.Timestamp(ws, tz="UTC")
    wet = pd.Timestamp(we, tz="UTC")
    out = []
    for t, s in trades:
        et = t.entry_time
        if et.tzinfo is None:
            et = et.tz_localize("UTC")
        if wst <= et <= wet:
            out.append((t, s))
    return out


def main():
    cache_dir = project_root / "data_cache"
    w = 140

    print("=" * w)
    print("MULTI-TIMEFRAME POSITION SIZING SIMULATION")
    print("=" * w)
    print(f"\n  Filter:  ADX>=25 + DI + BB mid + vol>=2.0x")
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  MaxConc: {MAX_CONCURRENT}  |  Start: ${START_EQUITY:,.0f}")

    # Collect trades with filter applied
    print(f"\n{'='*w}")
    print("PHASE 1: Collecting filtered trades (ADX>=25 + DI + BBmid + vol>=2.0)")
    print("=" * w)

    all_rows: list[tuple[LivePortfolioTrade, dict]] = []
    for sym in SYMBOLS:
        df_5m = load_candles(sym, cache_dir)
        if df_5m.empty:
            print(f"  SKIP {sym}", flush=True)
            continue
        for tf in TIMEFRAMES:
            t0 = time.time()
            ctx_df = build_bars_for_tf(df_5m, CONTEXT_TF.get(tf, "4h"))
            bars = build_bars_for_tf(df_5m, tf)
            chunk = collect_trades_live_aligned(
                bars, ctx_df, sym, tf, snap_fn=lambda b, c: _snap(b, c),
            )
            filtered = [(t, s) for t, s in chunk if _passes_filter(s, t.direction)]
            all_rows.extend(filtered)
            print(f"  {tf:>4s} {sym:<18s}: {len(chunk):>5d} raw -> {len(filtered):>4d} filtered  ({time.time()-t0:.1f}s)", flush=True)

    all_rows.sort(key=lambda x: x[0].entry_time)
    n = len(all_rows)

    tf_counts = {}
    for t, s in all_rows:
        tf_counts[t.tf] = tf_counts.get(t.tf, 0) + 1
    print(f"\n  Total filtered trades: {n:,}")
    for tf, cnt in sorted(tf_counts.items()):
        print(f"    {tf}: {cnt:,}")

    # Run each sizing scenario across each window
    print(f"\n{'='*w}")
    print("PHASE 2: RESULTS BY WINDOW AND SIZING")
    print("=" * w)

    for win_name, (ws, we) in WINDOWS.items():
        if win_name == "FULL":
            win_trades = all_rows
        else:
            win_trades = _filter_by_window(all_rows, ws, we)

        print(f"\n{'='*w}")
        print(f"  WINDOW: {win_name} ({ws} -> {we}) - {len(win_trades):,} trades")
        print(f"{'='*w}")

        hdr = (f"  {'Scenario':<28}  {'PnL':>9}  {'Ret%':>7}  {'MaxDD':>7}  "
               f"{'WR':>6}  {'Tr':>5}  {'15m':>5}  {'4h':>5}  "
               f"{'TP':>5}  {'SL':>5}  {'Trail':>5}  {'Fees':>8}")
        print(hdr)
        print("  " + "-" * (w - 4))

        for label, risk_map in SIZING_SCENARIOS:
            trades_list = [t for t, s in win_trades]
            r = run_portfolio_tf_sizing(
                trades_list, START_EQUITY, risk_map, MAX_CONCURRENT,
            )
            ret = r["pnl"] / START_EQUITY * 100
            tf_tr = r.get("tf_trades", {})
            print(
                f"  {label:<28}  ${r['pnl']:>+7,.0f}  {ret:>+6.0f}%  "
                f"{r['max_dd_pct']:>6.1%}  {r['wr']:>5.1f}%  {r['trades']:>5}  "
                f"{tf_tr.get('15m',0):>5}  {tf_tr.get('4h',0):>5}  "
                f"{r['tp']:>5}  {r['sl']:>5}  {r['trail']:>5}  ${r['total_fees']:>6,.0f}"
            )

    print(f"\n{'='*w}")
    print("SIMULATION COMPLETE")
    print("=" * w)


if __name__ == "__main__":
    main()
