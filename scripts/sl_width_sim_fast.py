"""
Optimised SL-width simulation.

Same outputs as `sl_width_sim.py`, but signals are collected ONCE per
(symbol, tf) and re-used across every (sl_mult, tp_mult) scenario.

Why it's correct:
    `sl_atr_mult` and `tp_atr_mult` only affect SL/TP **placement** in
    `SniperStrategy.generate_signal` (lines 230-240). They do not gate
    whether a signal fires. So per-scenario regeneration is wasted work.

Speedup vs. original:
    Original: 7-8 min per scenario × 14 scenarios ≈ 95 min.
    This:     ~30 s per scenario, signals cached → < 10 min total.

Other tweaks:
    - Monotonic context-bar pointer (`get_ctx`) — O(N+M) instead of O(N×M).
    - Cooldown is applied during *resolve* (per scenario), so different
      SL/TP combos correctly produce different cooldown windows.

Run via the wrapper that injects the live 6-symbol set:
    python scripts/_sim_overrides_fast.py
"""

from __future__ import annotations

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

# Sniper's class-level R:R gate (see strategies/strategy_sniper.py:44).
# Scenarios with rr < this would be rejected by `_finalize_signal` in
# production, so we drop them at resolve time too.
SNIPER_MIN_RR = 4.0

# When collecting signals we need to pick mults whose R:R passes the
# `_finalize_signal` gate. Use a permissive R:R; resolve enforces the
# real one.
COLLECT_SL_MULT = 1.0
COLLECT_TP_MULT = 10.0


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


# ── Phase 1 — collect signals once per (symbol, tf) ─────────────────
def collect_signals(trading_bars: pd.DataFrame, context_df: pd.DataFrame,
                    symbol: str, tf: str) -> list[dict]:
    """Return one dict per fired sniper signal.

    Keys: idx, ts, direction, entry_price, atr.
    sl_atr_mult/tp_atr_mult are dummies — they don't affect signal firing.
    """
    sniper = SniperStrategy(sl_atr_mult=COLLECT_SL_MULT,
                            tp_atr_mult=COLLECT_TP_MULT)

    if context_df is not None and not context_df.empty:
        # tolist() preserves tz; .values would strip it and break comparison
        # against the (tz-aware) trading-bar timestamps.
        ctx_ts = context_df["timestamp"].tolist()
        ctx_rows = [context_df.iloc[i] for i in range(len(context_df))]
    else:
        ctx_ts, ctx_rows = [], []

    ctx_pointer = [0]

    def get_ctx(ts: pd.Timestamp) -> pd.Series:
        if not ctx_ts:
            return pd.Series(dtype=object)
        p = ctx_pointer[0]
        while p + 1 < len(ctx_ts) and ctx_ts[p + 1] <= ts:
            p += 1
        ctx_pointer[0] = p
        return ctx_rows[p]

    n = len(trading_bars)
    timestamps = trading_bars["timestamp"].values

    signals: list[dict] = []
    for i in range(n):
        bar = trading_bars.iloc[i]
        ts = pd.Timestamp(bar.get("timestamp"))
        ctx = get_ctx(ts)
        sig = sniper.generate_signal(
            symbol=symbol, indicators_5m=bar,
            indicators_15m=ctx, funding_rate=0.0,
            liq_volume_1h=0.0,
        )
        if sig is None or sig.direction == "flat":
            continue

        atr_val = float(bar.get("atr_14") or 0.0)
        if atr_val <= 0 or sig.entry_price <= 0:
            continue

        signals.append({
            "idx": i,
            "ts": ts,
            "direction": sig.direction,
            "entry_price": float(sig.entry_price),
            "atr": atr_val,
        })

    return signals


# ── Phase 2 — resolve fast for any (sl_mult, tp_mult) combo ─────────
def resolve_trades(signals: list[dict], trading_bars: pd.DataFrame,
                   sl_mult: float, tp_mult: float,
                   symbol: str, tf: str) -> list[PrecomputedTrade]:
    """Walk forward each signal to determine exit. Apply COOLDOWN_BARS
    against this scenario's own exits (so wider SLs may keep different
    trades than tighter ones). Pure numpy on cached arrays."""
    if not signals:
        return []

    # Match sniper's _finalize_signal R:R gate. Scenarios that would be
    # rejected by the strategy in production produce 0 trades here.
    rr = tp_mult / sl_mult if sl_mult else 0
    if rr < SNIPER_MIN_RR:
        return []

    highs = trading_bars["high"].values.astype(np.float64)
    lows = trading_bars["low"].values.astype(np.float64)
    closes = trading_bars["close"].values.astype(np.float64)
    timestamps = trading_bars["timestamp"].values
    n = len(trading_bars)

    trades: list[PrecomputedTrade] = []
    last_exit_idx = -COOLDOWN_BARS - 1

    for sig in signals:
        i = sig["idx"]
        if i <= last_exit_idx + COOLDOWN_BARS:
            continue

        entry = sig["entry_price"]
        atr = sig["atr"]
        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr
        if sl_dist <= 0:
            continue

        direction = sig["direction"]
        if direction == "long":
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
            if direction == "long":
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

        if direction == "long":
            raw_pct = (exit_price - entry) / entry
        else:
            raw_pct = (entry - exit_price) / entry

        last_exit_idx = exit_idx
        trades.append(PrecomputedTrade(
            symbol=symbol, tf=tf, direction=direction,
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
    tp_count = sl_count = trail_count = timeout_count = 0

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
            elif trade.exit_reason == "timeout":
                timeout_count += 1

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
        "timeout": timeout_count,
    }


def load_candles(symbol: str, cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return pd.DataFrame()


def build_all_bars(cache_dir: Path) -> dict:
    """Pre-build trading/context bars for each (symbol, tf)."""
    bar_sets = {}
    for sym in SYMBOLS:
        df_5m = load_candles(sym, cache_dir)
        if df_5m.empty:
            print(f"  [skip] no cache for {sym}", flush=True)
            continue
        for tf in TIMEFRAMES:
            context_tf = CONTEXT_TF.get(tf, "4h")
            context_df = build_bars_for_tf(df_5m, context_tf)
            trading_bars = build_bars_for_tf(df_5m, tf)
            bar_sets[(sym, tf)] = (trading_bars, context_df)
    return bar_sets


def collect_all_signals(bar_sets: dict) -> dict:
    """One pass per (sym, tf). Returns dict[(sym,tf)] -> list[signal dict]."""
    out = {}
    for (sym, tf), (trading_bars, context_df) in bar_sets.items():
        t0 = time.time()
        sigs = collect_signals(trading_bars, context_df, sym, tf)
        out[(sym, tf)] = sigs
        print(f"  [signals] {sym} {tf:>3s}: {len(sigs):>5d} firings  "
              f"({time.time() - t0:.0f}s)", flush=True)
    return out


def run_scenario(bar_sets: dict, signals_per_symtf: dict,
                 sl_mult: float, tp_mult: float, label: str) -> dict:
    all_trades: list[PrecomputedTrade] = []
    for (sym, tf), (trading_bars, _ctx) in bar_sets.items():
        sigs = signals_per_symtf.get((sym, tf), [])
        trades = resolve_trades(sigs, trading_bars, sl_mult, tp_mult, sym, tf)
        all_trades.extend(trades)
    all_trades.sort(key=lambda t: t.entry_time)
    result = run_portfolio(all_trades)
    result["label"] = label
    result["sl_mult"] = sl_mult
    result["tp_mult"] = tp_mult
    result["rr"] = tp_mult / sl_mult if sl_mult else 0
    result["total_signals"] = len(all_trades)
    return result


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'SL':>5s}  {'TP':>5s}  {'R:R':>5s}  "
          f"{'Trades':>7s}  {'WR':>6s}  "
          f"{'Final Eq':>10s}  {'PnL':>12s}  {'Return':>8s}  "
          f"{'Max DD':>7s}  {'Min Eq':>9s}  "
          f"{'TP':>4s}  {'SL':>4s}  {'TR':>4s}  {'TO':>4s}", flush=True)
    print("-" * 120, flush=True)
    for r in rows:
        ret = r["pnl"] / START_EQUITY * 100
        print(f"{r['sl_mult']:>4.1f}x  {r['tp_mult']:>4.1f}x  "
              f"{r['rr']:>4.1f}:1  "
              f"{r['trades']:>7d}  {r['wr']:>5.1f}%  "
              f"${r['final_eq']:>8,.0f}  ${r['pnl']:>+10,.0f}  "
              f"{ret:>+7.1f}%  {r['max_dd_pct']:>6.1%}  "
              f"${r['min_equity']:>7,.0f}  "
              f"{r.get('tp', 0):>4d}  {r.get('sl', 0):>4d}  "
              f"{r.get('trail', 0):>4d}  {r.get('timeout', 0):>4d}",
              flush=True)


def main():
    cache_dir = project_root / "data_cache"

    print("=" * 90, flush=True)
    print("SL WIDTH SIM (FAST): wider SL = smaller position = more breathing room",
          flush=True)
    print(f"Equity: ${START_EQUITY:,.0f}  |  Risk: {RISK_PCT * 100:.2f}%  |  "
          f"Leverage: {LEVERAGE}x  |  Max concurrent: {MAX_CONCURRENT}",
          flush=True)
    print(f"Symbols: {SYMBOLS}", flush=True)
    print(f"Date range: {FROM} -> {TO}", flush=True)
    print("=" * 90, flush=True)

    print("\n-- Phase 1a: building bars --", flush=True)
    t0 = time.time()
    bar_sets = build_all_bars(cache_dir)
    print(f"   Done in {time.time() - t0:.1f}s  "
          f"({len(bar_sets)} symbol/tf combos)", flush=True)

    print("\n-- Phase 1b: collecting signals (one-time) --", flush=True)
    t0 = time.time()
    signals = collect_all_signals(bar_sets)
    total = sum(len(v) for v in signals.values())
    print(f"   Done in {time.time() - t0:.1f}s  "
          f"({total} signals total)", flush=True)

    # ── Mode A: Fixed R:R ──
    print("\n" + "=" * 90, flush=True)
    print(f"MODE A: Fixed R:R = {RR_RATIO}:1  (TP scales with SL, "
          f"position size shrinks)", flush=True)
    print("=" * 90, flush=True)

    results_a = []
    for sl_m in SL_MULTS_TO_TEST:
        tp_m = sl_m * RR_RATIO
        label = f"SL={sl_m:.1f}x / TP={tp_m:.1f}x"
        print(f"  Running {label} ...", end=" ", flush=True)
        t0 = time.time()
        r = run_scenario(bar_sets, signals, sl_m, tp_m, label)
        results_a.append(r)
        print(f"({time.time() - t0:.1f}s, {r['trades']} trades, "
              f"WR {r['wr']:.0f}%, ${r['pnl']:+,.0f})", flush=True)

    _print_table(results_a)

    # ── Mode B: Fixed TP, only SL widens ──
    print("\n" + "=" * 90, flush=True)
    print(f"MODE B: Fixed TP = {FIXED_TP_MULT}x ATR  (only SL widens, "
          f"R:R decreases as SL grows)", flush=True)
    print("=" * 90, flush=True)

    results_b = []
    for sl_m in SL_MULTS_TO_TEST:
        tp_m = FIXED_TP_MULT
        label = f"SL={sl_m:.1f}x / TP={tp_m:.1f}x"
        print(f"  Running {label} ...", end=" ", flush=True)
        t0 = time.time()
        r = run_scenario(bar_sets, signals, sl_m, tp_m, label)
        results_b.append(r)
        print(f"({time.time() - t0:.1f}s, {r['trades']} trades, "
              f"WR {r['wr']:.0f}%, ${r['pnl']:+,.0f})", flush=True)

    _print_table(results_b)

    best_a = max(results_a, key=lambda r: r["final_eq"])
    best_b = max(results_b, key=lambda r: r["final_eq"])
    print("\n" + "=" * 90, flush=True)
    print("BEST RESULTS (by Final Equity)", flush=True)
    print(f"  Mode A (fixed R:R):  {best_a['label']}  ->  "
          f"${best_a['final_eq']:,.0f}  (WR {best_a['wr']:.1f}%  "
          f"DD {best_a['max_dd_pct']:.1%})", flush=True)
    print(f"  Mode B (fixed TP):   {best_b['label']}  ->  "
          f"${best_b['final_eq']:,.0f}  (WR {best_b['wr']:.1f}%  "
          f"DD {best_b['max_dd_pct']:.1%})", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
