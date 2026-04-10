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
    run_portfolio_live_aligned,
    MAKER_FEE, TAKER_FEE, SLIPPAGE_BPS, LEVERAGE, NOTIONAL_CAP_FRAC,
    ROUND_TRIP_COST_SIZING, FUNDING_RATE_8H_FLAT,
    COOLDOWN_BARS, TRAIL_ACTIVATE, TRAIL_OFFSET,
    ENTRY_AGGRESSION_BPS, LIMIT_ORDER_TTL,
    _apply_exit_slippage, _funding_usd,
)
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.base import BaseStrategy, _sf, _valid

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

_CHOP_ADX_FLOOR = 25
_CHOP_VOL_FLOOR = 2.0

STRATEGIES_TO_TEST = [
    "sniper", "bb_squeeze", "rsi_divergence", "vwap_reversion",
    "multitf_scalp", "high_winrate", "volume_delta_liq",
]


def _passes_chop_filter(bar, direction):
    adx = _sf(bar.get("adx_14"))
    if adx > 0 and adx < _CHOP_ADX_FLOOR:
        return False
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
    return True


def collect_trades_for_strategy(
    strategy: BaseStrategy,
    trading_bars: pd.DataFrame,
    context_df: pd.DataFrame | None,
    symbol: str,
    tf: str,
    apply_chop_filter: bool = True,
) -> list[LivePortfolioTrade]:
    """Like collect_trades_live_aligned but accepts any BaseStrategy."""

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

    funding_col = None
    if "funding_8h" in trading_bars.columns:
        funding_col = trading_bars["funding_8h"].values

    out: list[LivePortfolioTrade] = []
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
                    bh, bl = highs[j], lows[j]
                    if sig.direction == "long":
                        gain = (bh - entry_fill) / entry_fill
                        if gain >= TRAIL_ACTIVATE:
                            trail_active = True
                        if trail_active:
                            if bh > hwm:
                                hwm = bh
                            new_sl = hwm * (1.0 - TRAIL_OFFSET)
                            if new_sl > current_sl:
                                current_sl = new_sl
                        sl_hit = bl <= current_sl
                        tp_hit = bh >= tp_abs
                    else:
                        gain = (entry_fill - bl) / entry_fill
                        if gain >= TRAIL_ACTIVATE:
                            trail_active = True
                        if trail_active:
                            if hwm == 0 or bl < hwm:
                                hwm = bl
                            new_sl = hwm * (1.0 + TRAIL_OFFSET)
                            if new_sl < current_sl:
                                current_sl = new_sl
                        sl_hit = bh >= current_sl
                        tp_hit = bl <= tp_abs

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
                    symbol=symbol,
                    tf=tf,
                    direction=sig.direction,
                    entry_time=pd.Timestamp(timestamps[k]),
                    exit_time=pd.Timestamp(timestamps[exit_idx]),
                    entry_price=entry_fill,
                    exit_price=exit_price,
                    sl_distance=sl_dist,
                    tp_distance=tp_dist,
                    raw_pct=raw_pct,
                    exit_reason=exit_reason,
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

        sig = strategy.generate_signal(
            symbol=symbol,
            indicators_5m=bar,
            indicators_15m=ctx,
            funding_rate=0.0,
            liq_volume_1h=0.0,
        )
        if sig is None or sig.direction == "flat":
            continue

        if apply_chop_filter and not _passes_chop_filter(bar, sig.direction):
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


def main():
    from src.strategies import STRATEGY_REGISTRY

    cache_dir = project_root / "data_cache"
    w = 140

    print("=" * w)
    print("STRATEGY COMPARISON SIMULATION")
    print("=" * w)
    print(f"  Filter:  ADX>=25 + DI + BB mid + vol>=2.0x (applied to all strategies)")
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

        for sym in precomputed:
            for tf in TIMEFRAMES:
                bars, ctx_df = precomputed[sym][tf]
                strat_instance = strat_cls()
                chunk = collect_trades_for_strategy(
                    strat_instance, bars, ctx_df, sym, tf,
                    apply_chop_filter=True,
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
