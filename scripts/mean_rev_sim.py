"""
MEAN-REVERSION STRATEGY SIMULATION

Tests the new MeanReversionStrategy on bear market windows:
  - Standalone (no chop filter - trades ADX < 20 conditions)
  - Combined with Sniper (Sniper uses chop filter, MR does not)

Also sweeps MR parameters: ADX ceiling, RSI thresholds, risk sizing.

Run:
    python scripts/mean_rev_sim.py
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
    run_portfolio_live_aligned,
    collect_trades_live_aligned,
    COOLDOWN_BARS, TRAIL_ACTIVATE, TRAIL_OFFSET,
    ENTRY_AGGRESSION_BPS, LIMIT_ORDER_TTL, SLIPPAGE_BPS,
    _apply_exit_slippage,
)
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.base import _sf, _valid
from src.strategies.strategy_mean_reversion import MeanReversionStrategy
from src.strategies.strategy_sniper import SniperStrategy

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


def _snap(bar, ctx):
    return {
        "adx_14": _sf(bar.get("adx_14")),
        "plus_di": _sf(bar.get("plus_di")),
        "minus_di": _sf(bar.get("minus_di")),
        "bb_mid": _sf(bar.get("bb_mid")),
        "volume_ratio": _sf(bar.get("volume_ratio")),
        "close": _sf(bar.get("close")),
    }


def _passes_chop_filter(snap, direction):
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


def collect_mr_trades(
    bars: pd.DataFrame,
    ctx_df: pd.DataFrame | None,
    symbol: str,
    tf: str,
    adx_ceil: float = 20,
    rsi_os: float = 30,
    rsi_ob: float = 70,
) -> list[LivePortfolioTrade]:
    """Collect trades using MeanReversionStrategy with configurable params."""

    strat = MeanReversionStrategy()
    strat.ADX_CEILING = adx_ceil
    strat.RSI_OVERSOLD = rsi_os
    strat.RSI_OVERBOUGHT = rsi_ob

    ctx_ts, ctx_rows = [], []
    if ctx_df is not None and not ctx_df.empty:
        for _, row in ctx_df.iterrows():
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

    highs = bars["high"].values.astype(np.float64)
    lows = bars["low"].values.astype(np.float64)
    closes = bars["close"].values.astype(np.float64)
    timestamps = bars["timestamp"].values
    n = len(bars)
    funding_col = bars["funding_8h"].values if "funding_8h" in bars.columns else None

    out: list[LivePortfolioTrade] = []
    last_exit_idx = -COOLDOWN_BARS - 1
    pending = None
    pending_bars = 0

    for i in range(n):
        bar = bars.iloc[i]
        ctx = get_ctx(pd.Timestamp(bar.get("timestamp")))

        if pending is not None:
            pending_bars += 1
            sig = pending["sig"]
            limit_px = pending["limit_px"]
            fr = pending["funding_rate_8h"]
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
                hwm = 0.0
                current_sl = sl_abs
                exit_price = None
                exit_reason = None
                exit_idx = n - 1

                for j in range(k + 1, n):
                    bh2, bl2 = highs[j], lows[j]
                    if sig.direction == "long":
                        gain = (bh2 - entry_fill) / entry_fill
                        if gain >= TRAIL_ACTIVATE:
                            trail_active = True
                        if trail_active:
                            if bh2 > hwm:
                                hwm = bh2
                            new_sl = hwm * (1.0 - TRAIL_OFFSET)
                            if new_sl > current_sl:
                                current_sl = new_sl
                        sl_hit = bl2 <= current_sl
                        tp_hit = bh2 >= tp_abs
                    else:
                        gain = (entry_fill - bl2) / entry_fill
                        if gain >= TRAIL_ACTIVATE:
                            trail_active = True
                        if trail_active:
                            if hwm == 0 or bl2 < hwm:
                                hwm = bl2
                            new_sl = hwm * (1.0 + TRAIL_OFFSET)
                            if new_sl < current_sl:
                                current_sl = new_sl
                        sl_hit = bh2 >= current_sl
                        tp_hit = bl2 <= tp_abs

                    if sl_hit:
                        exit_price = _apply_exit_slippage(
                            current_sl, sig.direction,
                            "trail" if trail_active else "sl",
                        )
                        exit_reason = "trail" if trail_active else "sl"
                        exit_idx = j
                        break
                    if tp_hit:
                        exit_price = _apply_exit_slippage(
                            tp_abs, sig.direction, "tp",
                        )
                        exit_reason = "tp"
                        exit_idx = j
                        break

                if exit_price is None:
                    exit_price = _apply_exit_slippage(
                        closes[-1], sig.direction, "timeout",
                    )
                    exit_reason = "timeout"
                    exit_idx = n - 1

                if sig.direction == "long":
                    raw_pct = (exit_price - entry_fill) / entry_fill
                else:
                    raw_pct = (entry_fill - exit_price) / entry_fill

                last_exit_idx = exit_idx
                pending = None
                pending_bars = 0
                sl_dist = abs(entry_fill - sl_abs)
                tp_dist = abs(tp_abs - entry_fill)

                trade = LivePortfolioTrade(
                    symbol=symbol, tf=tf, direction=sig.direction,
                    entry_time=pd.Timestamp(timestamps[k]),
                    exit_time=pd.Timestamp(timestamps[exit_idx]),
                    entry_price=entry_fill, exit_price=exit_price,
                    sl_distance=sl_dist, tp_distance=tp_dist,
                    raw_pct=raw_pct, exit_reason=exit_reason,
                    bars_held=exit_idx - k,
                    funding_rate_8h=fr,
                )
                out.append(trade)
                continue

            if pending_bars >= LIMIT_ORDER_TTL:
                pending = None
                pending_bars = 0
            continue

        if i <= last_exit_idx + COOLDOWN_BARS:
            continue

        sig = strat.generate_signal(
            symbol=symbol, indicators_5m=bar,
            indicators_15m=ctx, funding_rate=0.0, liq_volume_1h=0.0,
        )
        if sig is None or sig.direction == "flat":
            continue

        close = _sf(bar.get("close"))
        if close <= 0:
            continue

        if sig.direction == "long":
            limit_px = close * (1.0 + ENTRY_AGGRESSION_BPS)
        else:
            limit_px = close * (1.0 - ENTRY_AGGRESSION_BPS)

        fr = None
        if funding_col is not None and i < len(funding_col):
            v = funding_col[i]
            if v == v and float(v) != 0.0:
                fr = float(v)

        pending = {"sig": sig, "limit_px": limit_px, "funding_rate_8h": fr}
        pending_bars = 0

    return out


def load_candles(symbol, cache_dir):
    p = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


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


MR_PARAM_SWEEPS = [
    ("MR default (ADX<20, RSI 30/70)",    20, 30, 70, 0.010),
    ("MR wider RSI (ADX<20, RSI 35/65)",  20, 35, 65, 0.010),
    ("MR tight RSI (ADX<20, RSI 25/75)",  20, 25, 75, 0.010),
    ("MR ADX<25 RSI 30/70",               25, 30, 70, 0.010),
    ("MR ADX<25 RSI 35/65",               25, 35, 65, 0.010),
    ("MR ADX<15 RSI 30/70",               15, 30, 70, 0.010),
    ("MR default risk 1.5%",              20, 30, 70, 0.015),
    ("MR default risk 2.0%",              20, 30, 70, 0.020),
]


def main():
    cache_dir = project_root / "data_cache"
    w = 140

    print("=" * w)
    print("MEAN-REVERSION STRATEGY SIMULATION")
    print("=" * w)
    print(f"  Symbols: {', '.join(SYMBOLS)}")
    print(f"  MaxConc: {MAX_CONCURRENT}  |  Start: ${START_EQUITY:,.0f}")

    # Precompute bars
    print(f"\n{'='*w}")
    print("PHASE 1: Precomputing indicator bars")
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

    # Collect Sniper trades (with post-hoc chop filter for consistency)
    print(f"\n{'='*w}")
    print("PHASE 2: Collecting Sniper trades (with chop filter)")
    print("=" * w)

    sniper_trades: list[tuple[LivePortfolioTrade, dict]] = []
    for sym in precomputed:
        for tf in TIMEFRAMES:
            bars, ctx_df = precomputed[sym][tf]
            t0 = time.time()
            chunk = collect_trades_live_aligned(
                bars, ctx_df, sym, tf, snap_fn=lambda b, c: _snap(b, c),
            )
            filtered = [(t, s) for t, s in chunk if _passes_chop_filter(s, t.direction)]
            sniper_trades.extend(filtered)
            print(f"  {tf:>4s} {sym:<18s}: {len(chunk):>5d} raw -> {len(filtered):>4d} filtered  ({time.time()-t0:.1f}s)", flush=True)

    sniper_trades.sort(key=lambda x: x[0].entry_time)
    sniper_list = [t for t, s in sniper_trades]
    print(f"  Sniper total: {len(sniper_list):,} trades")

    # Collect MR trades for each parameter sweep
    print(f"\n{'='*w}")
    print("PHASE 3: Collecting MR trades (parameter sweep)")
    print("=" * w)

    mr_results: dict[str, list[LivePortfolioTrade]] = {}
    for label, adx_c, rsi_os, rsi_ob, risk in MR_PARAM_SWEEPS:
        t0_sweep = time.time()
        all_mr: list[LivePortfolioTrade] = []
        for sym in precomputed:
            for tf in TIMEFRAMES:
                bars, ctx_df = precomputed[sym][tf]
                chunk = collect_mr_trades(
                    bars, ctx_df, sym, tf,
                    adx_ceil=adx_c, rsi_os=rsi_os, rsi_ob=rsi_ob,
                )
                all_mr.extend(chunk)
        all_mr.sort(key=lambda t: t.entry_time)
        mr_results[label] = all_mr
        print(f"  {label:<40s}: {len(all_mr):>4d} trades  ({time.time()-t0_sweep:.1f}s)", flush=True)

    # Results
    print(f"\n{'='*w}")
    print("RESULTS: MR Standalone")
    print("=" * w)

    for win_name, (ws, we) in WINDOWS.items():
        print(f"\n  WINDOW: {win_name}")
        hdr = (f"  {'Config':<40}  {'PnL':>9}  {'Ret%':>7}  {'MaxDD':>7}  "
               f"{'WR':>6}  {'Tr':>5}  {'TP':>5}  {'SL':>5}")
        print(hdr)
        print("  " + "-" * (w - 4))

        for label, adx_c, rsi_os, rsi_ob, risk in MR_PARAM_SWEEPS:
            if win_name == "FULL":
                trades = mr_results[label]
            else:
                trades = _filter_by_window(mr_results[label], ws, we)
            if not trades:
                print(f"  {label:<40}  {'--':>9}  {'--':>7}  {'--':>7}  {'--':>6}  {'0':>5}")
                continue
            r = run_portfolio_live_aligned(
                trades, start_equity=START_EQUITY, risk_pct=risk,
                max_concurrent=MAX_CONCURRENT,
            )
            ret = r["pnl"] / START_EQUITY * 100
            print(
                f"  {label:<40}  ${r['pnl']:>+7,.0f}  {ret:>+6.0f}%  "
                f"{r['max_dd_pct']:>6.1%}  {r['wr']:>5.1f}%  {r['trades']:>5}  "
                f"{r['tp']:>5}  {r['sl']:>5}"
            )

    # Combined: Sniper + best MR
    print(f"\n{'='*w}")
    print("RESULTS: Sniper (2.5%) + MR Combined")
    print("=" * w)

    for win_name, (ws, we) in WINDOWS.items():
        print(f"\n  WINDOW: {win_name}")
        hdr = (f"  {'Config':<40}  {'PnL':>9}  {'Ret%':>7}  {'MaxDD':>7}  "
               f"{'WR':>6}  {'Tr':>5}  {'TP':>5}  {'SL':>5}")
        print(hdr)
        print("  " + "-" * (w - 4))

        # Sniper baseline (post-hoc filtered)
        if win_name == "FULL":
            sn_trades = sniper_list
        else:
            sn_trades = _filter_by_window(sniper_list, ws, we)
        if sn_trades:
            r = run_portfolio_live_aligned(
                sn_trades, start_equity=START_EQUITY, risk_pct=0.025,
                max_concurrent=MAX_CONCURRENT,
            )
            ret = r["pnl"] / START_EQUITY * 100
            print(
                f"  {'Sniper only (2.5%)':<40}  ${r['pnl']:>+7,.0f}  {ret:>+6.0f}%  "
                f"{r['max_dd_pct']:>6.1%}  {r['wr']:>5.1f}%  {r['trades']:>5}  "
                f"{r['tp']:>5}  {r['sl']:>5}"
            )

        for label, adx_c, rsi_os, rsi_ob, risk in MR_PARAM_SWEEPS:
            if win_name == "FULL":
                mr_trades = mr_results[label]
            else:
                mr_trades = _filter_by_window(mr_results[label], ws, we)
            combined = sorted(sn_trades + mr_trades, key=lambda t: t.entry_time)
            if not combined:
                continue
            r = run_portfolio_live_aligned(
                combined, start_equity=START_EQUITY, risk_pct=0.025,
                max_concurrent=MAX_CONCURRENT,
            )
            ret = r["pnl"] / START_EQUITY * 100
            print(
                f"  {'Sniper + ' + label:<40}  ${r['pnl']:>+7,.0f}  {ret:>+6.0f}%  "
                f"{r['max_dd_pct']:>6.1%}  {r['wr']:>5.1f}%  {r['trades']:>5}  "
                f"{r['tp']:>5}  {r['sl']:>5}"
            )

    print(f"\n{'='*w}")
    print("SIMULATION COMPLETE")
    print("=" * w)


if __name__ == "__main__":
    main()
