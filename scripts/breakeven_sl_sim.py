"""
BREAKEVEN STOP-LOSS SIMULATION

Tests moving SL to breakeven after the trade reaches X% profit,
specifically in choppy/bear market conditions.

Scenarios tested:
  A) Current setup: trail activates at 10%, offsets 3% from HWM
  B) BE@5%:  move SL to entry at +5%,  then trail at +10%
  C) BE@10%: move SL to entry at +10%, then trail at +10%
  D) BE@15%: move SL to entry at +15%, then trail at +15%
  E) BE@20%: move SL to entry at +20%, then trail at +20%
  F) BE@5%  only (no trailing stop)
  G) BE@10% only (no trailing stop)

Run:
    python scripts/breakeven_sl_sim.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.backtest.live_aligned_portfolio import (
    LivePortfolioTrade,
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
from src.strategies.strategy_sniper import SniperStrategy
from src.strategies.base import _sf

START_EQUITY = 3_000.0
RISK_PCT = 0.025
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

# Best bear filter from prior sim
_CHOP_ADX_FLOOR = 25
_CHOP_VOL_FLOOR = 2.0


def _snap(bar, ctx):
    return {
        "adx_14": _sf(bar.get("adx_14")),
        "plus_di": _sf(bar.get("plus_di")),
        "minus_di": _sf(bar.get("minus_di")),
        "bb_mid": _sf(bar.get("bb_mid")),
        "volume_ratio": _sf(bar.get("volume_ratio")),
        "close": _sf(bar.get("close")),
    }


def _passes_filter(snap: dict, direction: str) -> bool:
    adx = snap["adx_14"]
    if adx > 0 and adx < _CHOP_ADX_FLOOR:
        return False
    p, m = snap["plus_di"], snap["minus_di"]
    if p > 0 or m > 0:
        if direction == "long" and p <= m:
            return False
        if direction == "short" and m <= p:
            return False
    bb_mid = snap["bb_mid"]
    close = snap["close"]
    if bb_mid > 0 and close > 0:
        if direction == "long" and close <= bb_mid:
            return False
        if direction == "short" and close >= bb_mid:
            return False
    vr = snap["volume_ratio"]
    if vr > 0 and vr < _CHOP_VOL_FLOOR:
        return False
    return True


def _apply_exit_slippage(exit_price, direction, exit_reason):
    if exit_reason == "tp":
        return exit_price
    if direction == "long":
        return exit_price * (1.0 - SLIPPAGE_BPS)
    return exit_price * (1.0 + SLIPPAGE_BPS)


# ---------------------------------------------------------------------------
# Trade collector with configurable breakeven + trailing stop
# ---------------------------------------------------------------------------

def collect_trades_with_be(
    trading_bars: pd.DataFrame,
    context_df: pd.DataFrame | None,
    symbol: str,
    tf: str,
    be_activate_pct: float | None,
    trail_activate_pct: float | None,
    trail_offset_pct: float,
) -> list[tuple[LivePortfolioTrade, dict]]:
    """
    be_activate_pct:    move SL to entry when gain >= this (None = disabled)
    trail_activate_pct: activate trailing stop at this gain (None = disabled)
    trail_offset_pct:   trailing stop offset from HWM
    """
    sniper = SniperStrategy()
    out = []

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
    pending = None
    pending_bars = 0

    for i in range(n):
        bar = trading_bars.iloc[i]
        ctx = get_ctx(pd.Timestamp(bar.get("timestamp")))

        if pending is not None:
            pending_bars += 1
            sig = pending["sig"]
            limit_px = pending["limit_px"]
            snap = pending["snap"]
            bh, bl = highs[i], lows[i]
            filled = False
            fill_price = 0.0

            if sig.direction == "long":
                if bl <= limit_px:
                    fill_price = limit_px * (1.0 + SLIPPAGE_BPS)
                    filled = True
            else:
                if bh >= limit_px:
                    fill_price = limit_px * (1.0 - SLIPPAGE_BPS)
                    filled = True

            if filled:
                k = i
                entry_fill = fill_price
                sl_abs = float(sig.stop_loss)
                tp_abs = float(sig.take_profit)
                trail_active = False
                be_triggered = False
                hwm = 0.0
                current_sl = sl_abs
                exit_price = None
                exit_reason = None
                exit_idx = n - 1

                for j in range(k + 1, n):
                    bh, bl = highs[j], lows[j]
                    if sig.direction == "long":
                        gain = (bh - entry_fill) / entry_fill

                        if be_activate_pct is not None and not be_triggered and gain >= be_activate_pct:
                            be_triggered = True
                            if entry_fill > current_sl:
                                current_sl = entry_fill

                        if trail_activate_pct is not None and gain >= trail_activate_pct:
                            trail_active = True
                        if trail_active:
                            if bh > hwm:
                                hwm = bh
                            new_sl = hwm * (1.0 - trail_offset_pct)
                            if new_sl > current_sl:
                                current_sl = new_sl

                        sl_hit = bl <= current_sl
                        tp_hit = bh >= tp_abs
                    else:
                        gain = (entry_fill - bl) / entry_fill

                        if be_activate_pct is not None and not be_triggered and gain >= be_activate_pct:
                            be_triggered = True
                            if entry_fill < current_sl:
                                current_sl = entry_fill

                        if trail_activate_pct is not None and gain >= trail_activate_pct:
                            trail_active = True
                        if trail_active:
                            if hwm == 0 or bl < hwm:
                                hwm = bl
                            new_sl = hwm * (1.0 + trail_offset_pct)
                            if new_sl < current_sl:
                                current_sl = new_sl

                        sl_hit = bh >= current_sl
                        tp_hit = bl <= tp_abs

                    if sl_hit:
                        if trail_active:
                            ereason = "trail"
                        elif be_triggered:
                            ereason = "breakeven"
                        else:
                            ereason = "sl"
                        exit_price = _apply_exit_slippage(current_sl, sig.direction, ereason)
                        exit_reason = ereason
                        exit_idx = j
                        break
                    if tp_hit:
                        exit_price = _apply_exit_slippage(tp_abs, sig.direction, "tp")
                        exit_reason = "tp"
                        exit_idx = j
                        break

                if exit_price is None:
                    exit_price = _apply_exit_slippage(closes[-1], sig.direction, "timeout")
                    exit_reason = "timeout"
                    exit_idx = n - 1

                if sig.direction == "long":
                    raw_pct = (exit_price - entry_fill) / entry_fill
                else:
                    raw_pct = (entry_fill - exit_price) / entry_fill

                last_exit_idx = exit_idx
                pending = None
                pending_bars = 0

                trade = LivePortfolioTrade(
                    symbol=symbol, tf=tf, direction=sig.direction,
                    entry_time=pd.Timestamp(timestamps[k]),
                    exit_time=pd.Timestamp(timestamps[exit_idx]),
                    entry_price=entry_fill, exit_price=exit_price,
                    sl_distance=abs(entry_fill - sl_abs),
                    tp_distance=abs(tp_abs - entry_fill),
                    raw_pct=raw_pct, exit_reason=exit_reason,
                    bars_held=exit_idx - k,
                )
                out.append((trade, snap))
                continue

            if pending_bars >= LIMIT_ORDER_TTL:
                pending = None
                pending_bars = 0
            continue

        if i <= last_exit_idx + COOLDOWN_BARS:
            continue

        sig = sniper.generate_signal(
            symbol=symbol, indicators_5m=bar, indicators_15m=ctx,
            funding_rate=0.0, liq_volume_1h=0.0,
        )
        if sig is None or sig.direction == "flat":
            continue

        close = _sf(bar.get("close"))
        if close <= 0:
            continue

        snap = _snap(bar, ctx)
        if not _passes_filter(snap, sig.direction):
            continue

        if sig.direction == "long":
            limit_px = close * (1.0 + ENTRY_AGGRESSION_BPS)
        else:
            limit_px = close * (1.0 - ENTRY_AGGRESSION_BPS)

        pending = {"sig": sig, "limit_px": limit_px, "snap": snap}
        pending_bars = 0

    return out


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    ("A: Current (trail@10%, offset 3%)",
     None, 0.10, 0.03),

    ("B: BE@5% + trail@10%",
     0.05, 0.10, 0.03),

    ("C: BE@10% + trail@10%",
     0.10, 0.10, 0.03),

    ("D: BE@15% + trail@15%",
     0.15, 0.15, 0.03),

    ("E: BE@20% + trail@20%",
     0.20, 0.20, 0.03),

    ("F: BE@5% only (no trail)",
     0.05, None, 0.03),

    ("G: BE@10% only (no trail)",
     0.10, None, 0.03),

    ("H: BE@3% + trail@10%",
     0.03, 0.10, 0.03),

    ("I: BE@7% + trail@10%",
     0.07, 0.10, 0.03),
]


def load_candles(symbol, cache_dir):
    p = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


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
    print("BREAKEVEN STOP-LOSS SIMULATION")
    print("=" * w)
    print(f"\n  Filter:       ADX>=25 + DI + BB mid + vol>=2.0x")
    print(f"  Symbols:      {', '.join(SYMBOLS)}")
    print(f"  Risk:         {RISK_PCT:.1%}  |  MaxConc: {MAX_CONCURRENT}  |  Start: ${START_EQUITY:,.0f}")
    print(f"  Scenarios:    {len(SCENARIOS)}")
    print(f"  Windows:      {list(WINDOWS.keys())}")

    # Collect trades for each scenario
    print(f"\n{'='*w}")
    print("PHASE 1: Collecting trades for each breakeven/trail scenario")
    print("=" * w)

    scenario_trades: dict[str, list] = {}

    for label, be_pct, trail_pct, trail_off in SCENARIOS:
        print(f"\n  --- {label} ---")
        all_trades = []
        t0 = time.time()

        for sym in SYMBOLS:
            df_5m = load_candles(sym, cache_dir)
            if df_5m.empty:
                continue
            for tf in TIMEFRAMES:
                ctx_df = build_bars_for_tf(df_5m, CONTEXT_TF.get(tf, "4h"))
                bars = build_bars_for_tf(df_5m, tf)
                chunk = collect_trades_with_be(
                    bars, ctx_df, sym, tf,
                    be_activate_pct=be_pct,
                    trail_activate_pct=trail_pct,
                    trail_offset_pct=trail_off,
                )
                all_trades.extend(chunk)

        all_trades.sort(key=lambda x: x[0].entry_time)
        scenario_trades[label] = all_trades
        elapsed = time.time() - t0
        print(f"    {len(all_trades):,} trades collected ({elapsed:.0f}s)")

    # Run portfolio for each scenario x window
    print(f"\n{'='*w}")
    print("PHASE 2: RESULTS BY WINDOW")
    print("=" * w)

    for win_name, (ws, we) in WINDOWS.items():
        print(f"\n{'='*w}")
        print(f"  WINDOW: {win_name} ({ws} -> {we})")
        print(f"{'='*w}")

        hdr = (f"  {'Scenario':<35}  {'PnL':>9}  {'Ret%':>7}  {'MaxDD':>7}  "
               f"{'WR':>6}  {'Tr':>5}  {'TP':>5}  {'SL':>5}  {'Trail':>5}  "
               f"{'BE':>5}  {'T/O':>5}  {'Fees':>8}")
        print(hdr)
        print("  " + "-" * (w - 4))

        results_window = []
        for label, be_pct, trail_pct, trail_off in SCENARIOS:
            trades_all = scenario_trades[label]
            if win_name == "FULL":
                trades = [t for t, s in trades_all]
            else:
                trades = [t for t, s in _filter_by_window(trades_all, ws, we)]

            r = run_portfolio_live_aligned(
                trades, start_equity=START_EQUITY, risk_pct=RISK_PCT,
                max_concurrent=MAX_CONCURRENT,
            )

            be_count = sum(1 for t in trades if t.exit_reason == "breakeven")
            ret = r["pnl"] / START_EQUITY * 100

            results_window.append({
                "label": label, "pnl": r["pnl"], "ret": ret,
                "dd": r["max_dd_pct"], "wr": r["wr"],
                "trades": r["trades"], "tp": r["tp"], "sl": r["sl"],
                "trail": r["trail"], "be": be_count,
                "timeout": r["timeout"], "fees": r["total_fees"],
                "final_eq": r["final_eq"],
            })

            print(
                f"  {label:<35}  ${r['pnl']:>+7,.0f}  {ret:>+6.0f}%  "
                f"{r['max_dd_pct']:>6.1%}  {r['wr']:>5.1f}%  {r['trades']:>5}  "
                f"{r['tp']:>5}  {r['sl']:>5}  {r['trail']:>5}  "
                f"{be_count:>5}  {r['timeout']:>5}  ${r['total_fees']:>6,.0f}"
            )

    # Detailed exit breakdown
    print(f"\n{'='*w}")
    print("PHASE 3: EXIT REASON ANALYSIS (FULL period)")
    print("=" * w)

    for label, be_pct, trail_pct, trail_off in SCENARIOS:
        trades_all = scenario_trades[label]
        trades = [t for t, s in trades_all]
        total = len(trades)
        if total == 0:
            continue

        by_reason = defaultdict(list)
        for t in trades:
            by_reason[t.exit_reason].append(t)

        print(f"\n  {label}")
        print(f"  {'Reason':<12}  {'Count':>6}  {'%':>6}  {'AvgRaw%':>9}  {'Wins':>5}  {'WR':>6}")
        print(f"  {'-'*55}")
        for reason in ["tp", "trail", "breakeven", "sl", "timeout"]:
            ts = by_reason.get(reason, [])
            n = len(ts)
            if n == 0:
                continue
            avg = sum(t.raw_pct for t in ts) / n * 100
            wins = sum(1 for t in ts if t.raw_pct > 0)
            wr = wins / n * 100
            print(f"  {reason:<12}  {n:>6}  {n/total*100:>5.1f}%  {avg:>+8.2f}%  {wins:>5}  {wr:>5.1f}%")

    # Per-symbol impact of breakeven
    print(f"\n{'='*w}")
    print("PHASE 4: PER-SYMBOL COMPARISON (Current vs Best BE config, BEAR-2023)")
    print("=" * w)

    current_label = SCENARIOS[0][0]
    best_be_label = None
    best_be_pnl = -999999

    for label, be_pct, trail_pct, trail_off in SCENARIOS:
        if be_pct is not None:
            trades_all = scenario_trades[label]
            bear_trades = [t for t, s in _filter_by_window(trades_all, "2023-01-01", "2023-12-31")]
            r = run_portfolio_live_aligned(
                bear_trades, start_equity=START_EQUITY, risk_pct=RISK_PCT,
                max_concurrent=MAX_CONCURRENT,
            )
            if r["pnl"] > best_be_pnl:
                best_be_pnl = r["pnl"]
                best_be_label = label

    if best_be_label:
        print(f"\n  Best BE config for BEAR-2023: {best_be_label} (PnL: ${best_be_pnl:+,.0f})")

        for label in [current_label, best_be_label]:
            trades_all = scenario_trades[label]
            bear_trades_with_snap = _filter_by_window(trades_all, "2023-01-01", "2023-12-31")

            print(f"\n  --- {label} ---")
            print(f"  {'Symbol':<18}  {'Trades':>7}  {'Wins':>5}  {'WR':>6}  {'AvgRaw%':>9}")
            print(f"  {'-'*55}")

            by_sym = defaultdict(list)
            for t, s in bear_trades_with_snap:
                by_sym[t.symbol].append(t)

            for sym in SYMBOLS:
                ts = by_sym.get(sym, [])
                n = len(ts)
                if n == 0:
                    print(f"  {sym:<18}  {0:>7}  {0:>5}  {'N/A':>6}  {'N/A':>9}")
                    continue
                wins = sum(1 for t in ts if t.raw_pct > 0)
                avg = sum(t.raw_pct for t in ts) / n * 100
                print(f"  {sym:<18}  {n:>7}  {wins:>5}  {wins/n*100:>5.1f}%  {avg:>+8.2f}%")

    print(f"\n{'='*w}")
    print("SIMULATION COMPLETE")
    print("=" * w)


if __name__ == "__main__":
    main()
