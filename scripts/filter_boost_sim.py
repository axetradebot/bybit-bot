"""
Filter Boost Ablation Study for Sniper Strategy.

FAST version: runs a single pass per (symbol, tf) to collect all baseline
signals, then replays each filter/combo using vectorised numpy exit scanning.

Usage
-----
    python scripts/filter_boost_sim.py
    python scripts/filter_boost_sim.py --symbols SOLUSDT DOGEUSDT
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import structlog

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.backtest.simulator import Simulator, MAKER_FEE
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.strategy_sniper import SniperStrategy
from src.strategies.base import _sf, _valid

log = structlog.get_logger()

DEFAULT_SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT", "DOTUSDT",
    "1000PEPEUSDT", "DOGEUSDT", "OPUSDT", "NEARUSDT", "SUIUSDT",
]
TIMEFRAMES = ["15m", "4h"]
EQUITY = 10_000.0
RISK_PCT = 0.02
LEVERAGE = 10
COOLDOWN = 6

Simulator._load_funding = lambda self, engine: {}


# -----------------------------------------------------------------------
# Data fetching
# -----------------------------------------------------------------------

def fetch_5m_candles(symbol: str, since: str, until: str) -> pd.DataFrame:
    import ccxt
    exchange = ccxt.bybit({"enableRateLimit": True})
    exchange.load_markets()
    from src.live.order_manager import _to_ccxt_symbol
    bybit_symbol = _to_ccxt_symbol(symbol)
    if bybit_symbol not in exchange.markets:
        bybit_symbol = symbol.replace("USDT", "/USDT:USDT")
    since_ts = int(datetime.fromisoformat(since).replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    until_ts = int(datetime.fromisoformat(until).replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    all_candles: list[list] = []
    current = since_ts
    while current < until_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(
                bybit_symbol, "5m", since=current, limit=1000)
        except Exception as e:
            print(f"  [WARN] {symbol} fetch error: {e}", flush=True)
            time.sleep(2)
            continue
        if not ohlcv:
            break
        for candle in ohlcv:
            if candle[0] >= until_ts:
                break
            all_candles.append(candle)
        last_ts = ohlcv[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 1
    if not all_candles:
        return pd.DataFrame()
    df = pd.DataFrame(all_candles,
                      columns=["timestamp", "open", "high", "low",
                               "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (df.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp").reset_index(drop=True))
    df["symbol"] = symbol
    df["buy_volume"] = df["volume"] * 0.5
    df["sell_volume"] = df["volume"] * 0.5
    df["volume_delta"] = 0.0
    df["quote_volume"] = df["volume"] * df["close"]
    df["trade_count"] = 0
    df["mark_price"] = df["close"]
    df["funding_rate"] = 0.0
    return df


# -----------------------------------------------------------------------
# Signal record
# -----------------------------------------------------------------------

@dataclass
class SignalRecord:
    bar_idx: int
    direction: str      # "long" / "short"
    entry_price: float
    sl_distance: float
    tp_distance: float
    bar_data: dict
    ctx_data: dict
    prev_macd_hist: float


# -----------------------------------------------------------------------
# Numpy-backed bar arrays for fast exit scanning
# -----------------------------------------------------------------------

@dataclass
class BarArrays:
    highs: np.ndarray
    lows: np.ndarray
    n: int


def make_bar_arrays(df: pd.DataFrame) -> BarArrays:
    return BarArrays(
        highs=df["high"].values.astype(np.float64),
        lows=df["low"].values.astype(np.float64),
        n=len(df),
    )


def find_exit(ba: BarArrays, entry_idx: int, direction: str,
              sl: float, tp: float) -> tuple[int, str, float]:
    """Scan forward from entry_idx+1 to find first bar hitting SL or TP.
    Returns (exit_bar_idx, reason, exit_price)."""
    start = entry_idx + 1
    if direction == "long":
        for i in range(start, ba.n):
            if ba.lows[i] <= sl:
                return i, "sl", sl
            if ba.highs[i] >= tp:
                return i, "tp", tp
    else:
        for i in range(start, ba.n):
            if ba.highs[i] >= sl:
                return i, "sl", sl
            if ba.lows[i] <= tp:
                return i, "tp", tp
    return -1, "open", 0.0


# -----------------------------------------------------------------------
# Fast trade replay using numpy arrays
# -----------------------------------------------------------------------

def replay_trades(signals: list[SignalRecord], ba: BarArrays) -> dict:
    if not signals:
        return {"pnl": 0.0, "trades": 0, "wins": 0, "tp": 0, "sl": 0}

    total_pnl = 0.0
    trades = 0
    wins = 0
    tp_count = 0
    sl_count = 0
    notional = EQUITY * RISK_PCT * LEVERAGE
    fee_per_side = notional * MAKER_FEE

    last_exit_idx = -COOLDOWN - 1

    for sig in signals:
        if sig.bar_idx - last_exit_idx < COOLDOWN:
            continue
        if sig.bar_idx <= last_exit_idx:
            continue

        ep = sig.entry_price
        if sig.direction == "long":
            sl = ep - sig.sl_distance
            tp = ep + sig.tp_distance
        else:
            sl = ep + sig.sl_distance
            tp = ep - sig.tp_distance

        exit_idx, reason, exit_price = find_exit(
            ba, sig.bar_idx, sig.direction, sl, tp)

        if exit_idx < 0:
            continue

        if sig.direction == "long":
            pnl_pct = (exit_price - ep) / ep
        else:
            pnl_pct = (ep - exit_price) / ep

        pnl_pct *= LEVERAGE
        pnl_usd = notional * (pnl_pct / LEVERAGE) - 2 * fee_per_side

        total_pnl += pnl_usd
        trades += 1
        if pnl_usd > 0:
            wins += 1
        if reason == "tp":
            tp_count += 1
        else:
            sl_count += 1

        last_exit_idx = exit_idx

    return {
        "pnl": total_pnl, "trades": trades, "wins": wins,
        "tp": tp_count, "sl": sl_count,
    }


# -----------------------------------------------------------------------
# Collect baseline signals
# -----------------------------------------------------------------------

def collect_signals(
    symbol: str, trading_bars: pd.DataFrame, context_df: pd.DataFrame,
) -> list[SignalRecord]:
    sniper = SniperStrategy()
    signals: list[SignalRecord] = []

    ctx_ts = []
    ctx_rows = []
    if context_df is not None and not context_df.empty:
        for _, row in context_df.iterrows():
            ctx_ts.append(pd.Timestamp(row["timestamp"]))
            ctx_rows.append(row)

    def get_ctx(bar_ts):
        if not ctx_ts:
            return pd.Series(dtype=object)
        idx = 0
        for i, t in enumerate(ctx_ts):
            if t <= bar_ts:
                idx = i
            else:
                break
        return ctx_rows[idx]

    prev_macd = 0.0
    for i in range(len(trading_bars)):
        bar = trading_bars.iloc[i]
        ctx = get_ctx(pd.Timestamp(bar.get("timestamp")))
        cur_macd = _sf(bar.get("macd_hist"))
        sig = sniper.generate_signal(
            symbol=symbol, indicators_5m=bar,
            indicators_15m=ctx, funding_rate=0.0, liq_volume_1h=0.0,
        )
        if sig is not None:
            close = float(bar["close"])
            signals.append(SignalRecord(
                bar_idx=i,
                direction=sig.direction,
                entry_price=close,
                sl_distance=abs(close - sig.stop_loss),
                tp_distance=abs(sig.take_profit - close),
                bar_data=bar.to_dict(),
                ctx_data=ctx.to_dict() if hasattr(ctx, 'to_dict') else {},
                prev_macd_hist=prev_macd,
            ))
        prev_macd = cur_macd

    return signals


# -----------------------------------------------------------------------
# Filter functions: (bar_data, ctx_data, direction) -> bool
# -----------------------------------------------------------------------

def f_adx_20(b, c, d): return _sf(b.get("adx_14")) >= 20
def f_adx_25(b, c, d): return _sf(b.get("adx_14")) >= 25
def f_adx_30(b, c, d): return _sf(b.get("adx_14")) >= 30

def f_adx_di(b, c, d):
    p, m = _sf(b.get("plus_di")), _sf(b.get("minus_di"))
    return (p > m) if d == "long" else (m > p) if (p or m) else False

def f_volume_spike(b, c, d): return _sf(b.get("volume_ratio")) > 1.2

def f_mfi_zone(b, c, d):
    mfi = _sf(b.get("mfi_14"))
    return 30 <= mfi <= 70

def f_obv_trend(b, c, d):
    s = _sf(b.get("obv_slope"))
    return (s > 0) if d == "long" else (s < 0)

def f_order_flow(b, c, d):
    o = _sf(b.get("order_flow_imb"))
    return (o > 0.52) if d == "long" else (o < 0.48)

def f_cmf_confirm(b, c, d):
    cmf = _sf(b.get("cmf_20"))
    return (cmf > 0) if d == "long" else (cmf < 0)

def f_macd_hist_dir(b, c, d):
    h = _sf(b.get("macd_hist"))
    return (h > 0) if d == "long" else (h < 0)

def f_macd_accel(b, c, d):
    h = _sf(b.get("macd_hist"))
    p = b.get("_prev_macd_hist", h)
    return (h > p) if d == "long" else (h < p)

def f_stochrsi_zone(b, c, d):
    k = _sf(b.get("stochrsi_k"))
    return 20 <= k <= 80

def f_vwap_side(b, c, d):
    cl, vw = _sf(b.get("close")), _sf(b.get("vwap"))
    if vw <= 0: return True
    return (cl > vw) if d == "long" else (cl < vw)

def f_bb_position(b, c, d):
    cl, bm = _sf(b.get("close")), _sf(b.get("bb_mid"))
    if bm <= 0: return True
    return (cl < bm) if d == "long" else (cl > bm)

def f_candle_body(b, c, d): return _sf(b.get("candle_body_ratio")) > 0.5

def f_ctx_rsi(b, c, d):
    r = _sf(c.get("rsi_14"))
    return 40 <= r <= 60 if r else True

def f_ctx_adx(b, c, d):
    a = _sf(c.get("adx_14"))
    return a > 20 if a else True

def f_no_divergence(b, c, d):
    ext = b.get("extras") or {}
    return (not ext.get("div_regular_bear", False) if d == "long"
            else not ext.get("div_regular_bull", False))

def f_willr_zone(b, c, d):
    w = _sf(b.get("willr_14"))
    return (w < -30) if d == "long" else (w > -70)

def f_roc_confirm(b, c, d):
    r = _sf(b.get("roc_10"))
    return (r > 0) if d == "long" else (r < 0)


FILTERS: dict[str, Callable] = {
    "adx>=20":       f_adx_20,
    "adx>=25":       f_adx_25,
    "adx>=30":       f_adx_30,
    "adx_di":        f_adx_di,
    "vol_spike":     f_volume_spike,
    "mfi_zone":      f_mfi_zone,
    "obv_trend":     f_obv_trend,
    "order_flow":    f_order_flow,
    "cmf_confirm":   f_cmf_confirm,
    "macd_hist_dir": f_macd_hist_dir,
    "macd_accel":    f_macd_accel,
    "stochrsi_zone": f_stochrsi_zone,
    "vwap_side":     f_vwap_side,
    "bb_position":   f_bb_position,
    "candle_body":   f_candle_body,
    "ctx_rsi":       f_ctx_rsi,
    "ctx_adx":       f_ctx_adx,
    "no_divergence": f_no_divergence,
    "willr_zone":    f_willr_zone,
    "roc_confirm":   f_roc_confirm,
}


# -----------------------------------------------------------------------
# Apply filters to signal list
# -----------------------------------------------------------------------

def apply_filters(sigs: list[SignalRecord], fnames: list[str]) -> list[SignalRecord]:
    fns = [FILTERS[f] for f in fnames]
    out = []
    for s in sigs:
        bd = dict(s.bar_data)
        bd["_prev_macd_hist"] = s.prev_macd_hist
        if all(fn(bd, s.ctx_data, s.direction) for fn in fns):
            out.append(s)
    return out


# -----------------------------------------------------------------------
# Mean-Reversion signal collector
# -----------------------------------------------------------------------

def collect_mr_signals(trading_bars: pd.DataFrame) -> list[SignalRecord]:
    sigs = []
    n = len(trading_bars)
    close_arr = trading_bars["close"].values.astype(np.float64)
    atr_arr = trading_bars["atr_14"].values.astype(np.float64) if "atr_14" in trading_bars else np.zeros(n)
    rsi_arr = trading_bars["rsi_14"].values.astype(np.float64) if "rsi_14" in trading_bars else np.full(n, 50.0)
    mfi_arr = trading_bars["mfi_14"].values.astype(np.float64) if "mfi_14" in trading_bars else np.full(n, 50.0)
    bbl_arr = trading_bars["bb_lower"].values.astype(np.float64) if "bb_lower" in trading_bars else np.full(n, 0.0)
    bbu_arr = trading_bars["bb_upper"].values.astype(np.float64) if "bb_upper" in trading_bars else np.full(n, 1e18)
    hac_arr = trading_bars["ha_close"].values.astype(np.float64) if "ha_close" in trading_bars else close_arr
    hao_arr = trading_bars["ha_open"].values.astype(np.float64) if "ha_open" in trading_bars else close_arr

    for i in range(1, n):
        cl = close_arr[i]
        atr = atr_arr[i]
        if atr <= 0 or cl <= 0 or np.isnan(atr) or np.isnan(cl):
            continue

        rsi = rsi_arr[i]; mfi = mfi_arr[i]
        if np.isnan(rsi) or np.isnan(mfi):
            continue

        direction = None
        if cl <= bbl_arr[i] and rsi < 30 and mfi < 35:
            if hac_arr[i] > hao_arr[i]:
                direction = "long"
        elif cl >= bbu_arr[i] and rsi > 70 and mfi > 65:
            if hac_arr[i] < hao_arr[i]:
                direction = "short"

        if direction is None:
            continue

        sigs.append(SignalRecord(
            bar_idx=i, direction=direction, entry_price=cl,
            sl_distance=1.5*atr, tp_distance=3.0*atr,
            bar_data={}, ctx_data={}, prev_macd_hist=0,
        ))
    return sigs


# -----------------------------------------------------------------------
# Aggregate helper
# -----------------------------------------------------------------------

def agg_add(agg: dict, r: dict):
    agg["pnl"] += r["pnl"]
    agg["trades"] += r["trades"]
    agg["wins"] += r["wins"]
    agg["tp"] += r["tp"]
    agg["sl"] += r["sl"]

def new_agg() -> dict:
    return {"pnl": 0.0, "trades": 0, "wins": 0, "tp": 0, "sl": 0}


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Filter Boost Ablation Study")
    p.add_argument("--from", dest="from_date", default="2023-01-01")
    p.add_argument("--to", dest="to_date",
                   default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    p.add_argument("--top-n-combos", type=int, default=10)
    args = p.parse_args()

    fnames = list(FILTERS.keys())

    print(f"\n{'='*110}")
    print(f"  FILTER BOOST ABLATION STUDY (fast mode)")
    print(f"  {args.from_date} -> {args.to_date}")
    print(f"  Symbols:    {', '.join(args.symbols)}")
    print(f"  Timeframes: {', '.join(args.timeframes)}")
    print(f"  Filters:    {len(fnames)}")
    print(f"  Equity: ${EQUITY:,.0f}  |  Risk: {RISK_PCT:.0%}  |  Leverage: {LEVERAGE}x")
    print(f"{'='*110}\n")

    t0 = time.time()

    # ── Load data ────────────────────────────────────────────
    cache_dir = project_root / "data_cache"
    cache_dir.mkdir(exist_ok=True)

    candle_cache: dict[str, pd.DataFrame] = {}
    for idx, sym in enumerate(args.symbols):
        cache_path = cache_dir / f"{sym}_{args.from_date}_{args.to_date}_5m.parquet"
        if cache_path.exists():
            print(f"[{idx+1}/{len(args.symbols)}] {sym} -- cache ...",
                  end=" ", flush=True)
            df = pd.read_parquet(cache_path)
        else:
            print(f"[{idx+1}/{len(args.symbols)}] {sym} -- downloading ...",
                  end=" ", flush=True)
            df = fetch_5m_candles(sym, args.from_date, args.to_date)
            if not df.empty:
                df.to_parquet(cache_path)
        print(f"{len(df):,} candles", flush=True)
        candle_cache[sym] = df

    # ── Build bars + collect signals + numpy arrays ──────────
    print(f"\nBuilding bars & collecting signals ...", flush=True)

    datasets: list[tuple[str, str, list[SignalRecord], BarArrays, pd.DataFrame]] = []

    for sym in args.symbols:
        df_5m = candle_cache[sym]
        if df_5m.empty:
            continue
        for tf in args.timeframes:
            trading_bars = build_bars_for_tf(df_5m, tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            context_df = (build_bars_for_tf(df_5m, ctx_tf)
                          if ctx_tf != tf else trading_bars)
            sigs = collect_signals(sym, trading_bars, context_df)
            ba = make_bar_arrays(trading_bars)
            datasets.append((sym, tf, sigs, ba, trading_bars))
            print(f"  {sym} {tf}: {len(sigs)} signals from "
                  f"{ba.n:,} bars", flush=True)

    t_collect = time.time()
    print(f"  Signal collection: {t_collect - t0:.0f}s", flush=True)

    # ── Phase 1: Baseline ────────────────────────────────────
    print(f"\n--- Phase 1: Baseline Sniper ---", flush=True)
    bl = new_agg()
    for sym, tf, sigs, ba, _ in datasets:
        agg_add(bl, replay_trades(sigs, ba))
    bl_wr = bl["wins"] / bl["trades"] if bl["trades"] else 0
    print(f"  Baseline: {bl['trades']} trades, WR {bl_wr:.1%}, "
          f"PnL ${bl['pnl']:+,.2f}")

    # ── Phase 2: Individual filters ──────────────────────────
    print(f"\n--- Phase 2: Individual Filters ({len(fnames)}) ---", flush=True)
    filter_results: dict[str, dict] = {}

    for fi, fn in enumerate(fnames):
        ag = new_agg()
        for sym, tf, sigs, ba, _ in datasets:
            filtered = apply_filters(sigs, [fn])
            agg_add(ag, replay_trades(filtered, ba))
        wr = ag["wins"] / ag["trades"] if ag["trades"] else 0
        delta = ag["pnl"] - bl["pnl"]
        filter_results[fn] = {**ag, "wr": wr, "delta": delta}
        print(f"  [{fi+1:>2}/{len(fnames)}] {fn:<18} "
              f"{ag['trades']:>6} trades  WR {wr:>5.1%}  "
              f"PnL ${ag['pnl']:>+11,.2f}  delta ${delta:>+10,.2f}", flush=True)

    # ── Phase 3: Combinations ────────────────────────────────
    print(f"\n--- Phase 3: Filter Combinations ---", flush=True)

    positive = [f for f, r in filter_results.items() if r["delta"] > 0]
    print(f"  Filters with positive delta: {len(positive)}")
    if len(positive) < 2:
        positive = sorted(filter_results,
                          key=lambda f: filter_results[f]["delta"],
                          reverse=True)[:8]
        print(f"  Using top 8 by delta: {positive}")

    combos = []
    for r in range(2, min(4, len(positive) + 1)):
        combos.extend(itertools.combinations(positive, r))
    print(f"  Testing {len(combos)} combinations ...", flush=True)

    combo_results: list[tuple[tuple[str, ...], dict]] = []
    for ci, combo in enumerate(combos):
        ag = new_agg()
        for sym, tf, sigs, ba, _ in datasets:
            filtered = apply_filters(sigs, list(combo))
            agg_add(ag, replay_trades(filtered, ba))
        wr = ag["wins"] / ag["trades"] if ag["trades"] else 0
        delta = ag["pnl"] - bl["pnl"]
        combo_results.append((combo, {**ag, "wr": wr, "delta": delta}))
        if (ci+1) % 20 == 0 or ci == len(combos) - 1:
            print(f"    {ci+1}/{len(combos)} done", flush=True)

    # ── Phase 4: Strategy Variants ───────────────────────────
    print(f"\n--- Phase 4: Strategy Variants ---", flush=True)

    # Sniper Momentum
    mom = new_agg()
    for sym, tf, sigs, ba, _ in datasets:
        filtered = apply_filters(sigs, ["adx>=25", "macd_hist_dir", "vol_spike"])
        agg_add(mom, replay_trades(filtered, ba))
    mom_wr = mom["wins"] / mom["trades"] if mom["trades"] else 0
    print(f"  Sniper Momentum: {mom['trades']} trades, WR {mom_wr:.1%}, "
          f"PnL ${mom['pnl']:+,.2f}")

    # Mean-Reversion
    mr = new_agg()
    for sym, tf, _, ba, bars in datasets:
        mr_sigs = collect_mr_signals(bars)
        agg_add(mr, replay_trades(mr_sigs, ba))
    mr_wr = mr["wins"] / mr["trades"] if mr["trades"] else 0
    print(f"  Mean-Reversion:  {mr['trades']} trades, WR {mr_wr:.1%}, "
          f"PnL ${mr['pnl']:+,.2f}")

    # ═══════════════════════════════════════════════════════════
    #  RESULTS
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'='*110}")
    print(f"  INDIVIDUAL FILTER RESULTS (ranked by PnL delta)")
    print(f"{'='*110}")
    hdr = (f"{'Rank':>4} {'Filter':<18} {'Trades':>7} {'WR':>6} "
           f"{'PnL $':>12} {'Delta $':>11} {'TP':>5} {'SL':>5}")
    print(hdr)
    print("-" * len(hdr))

    ranked = sorted(filter_results.items(),
                    key=lambda x: x[1]["delta"], reverse=True)
    for i, (fn, r) in enumerate(ranked):
        print(f"{i+1:>4} {fn:<18} {r['trades']:>7} {r['wr']:>6.1%} "
              f"{r['pnl']:>+12,.2f} {r['delta']:>+11,.2f} "
              f"{r['tp']:>5} {r['sl']:>5}")

    print(f"\n  Baseline: {bl['trades']} trades, WR {bl_wr:.1%}, "
          f"PnL ${bl['pnl']:>+12,.2f}")

    # Top combos
    print(f"\n{'='*110}")
    print(f"  TOP {args.top_n_combos} FILTER COMBINATIONS (by PnL)")
    print(f"{'='*110}")
    combo_ranked = sorted(combo_results, key=lambda x: x[1]["pnl"],
                          reverse=True)[:args.top_n_combos]
    hdr2 = (f"{'Rank':>4} {'Combination':<40} {'Trades':>7} {'WR':>6} "
            f"{'PnL $':>12} {'Delta $':>11}")
    print(hdr2)
    print("-" * len(hdr2))
    for i, (combo, r) in enumerate(combo_ranked):
        label = " + ".join(combo)
        delta = r["pnl"] - bl["pnl"]
        print(f"{i+1:>4} {label:<40} {r['trades']:>7} {r['wr']:>6.1%} "
              f"{r['pnl']:>+12,.2f} {delta:>+11,.2f}")

    # Strategy variants
    print(f"\n{'='*110}")
    print(f"  STRATEGY VARIANT COMPARISON")
    print(f"{'='*110}")
    print(f"  {'Baseline Sniper:':<25} {bl['trades']:>6} trades  "
          f"WR {bl_wr:>5.1%}  PnL ${bl['pnl']:>+12,.2f}")
    print(f"  {'Sniper Momentum:':<25} {mom['trades']:>6} trades  "
          f"WR {mom_wr:>5.1%}  PnL ${mom['pnl']:>+12,.2f}  "
          f"delta ${mom['pnl']-bl['pnl']:>+10,.2f}")
    print(f"  {'Mean-Reversion:':<25} {mr['trades']:>6} trades  "
          f"WR {mr_wr:>5.1%}  PnL ${mr['pnl']:>+12,.2f}  "
          f"delta ${mr['pnl']-bl['pnl']:>+10,.2f}")

    # Per-symbol breakdown for top 3 filters + strategies
    print(f"\n{'='*110}")
    print(f"  PER-SYMBOL BREAKDOWN")
    print(f"{'='*110}")
    top3 = [fn for fn, _ in ranked[:3]]

    for fn in top3:
        print(f"\n  Filter: {fn}")
        print(f"    {'Symbol':<16} {'TF':>4} {'Trades':>7} {'WR':>6} {'PnL $':>12}")
        for sym, tf, sigs, ba, _ in datasets:
            filtered = apply_filters(sigs, [fn])
            r = replay_trades(filtered, ba)
            wr = r["wins"]/r["trades"] if r["trades"] else 0
            print(f"    {sym:<16} {tf:>4} {r['trades']:>7} {wr:>6.1%} "
                  f"{r['pnl']:>+12,.2f}")

    print(f"\n  Sniper Momentum (per-symbol):")
    print(f"    {'Symbol':<16} {'TF':>4} {'Trades':>7} {'WR':>6} {'PnL $':>12}")
    for sym, tf, sigs, ba, _ in datasets:
        filtered = apply_filters(sigs, ["adx>=25", "macd_hist_dir", "vol_spike"])
        r = replay_trades(filtered, ba)
        wr = r["wins"]/r["trades"] if r["trades"] else 0
        print(f"    {sym:<16} {tf:>4} {r['trades']:>7} {wr:>6.1%} "
              f"{r['pnl']:>+12,.2f}")

    print(f"\n  Mean-Reversion (per-symbol):")
    print(f"    {'Symbol':<16} {'TF':>4} {'Trades':>7} {'WR':>6} {'PnL $':>12}")
    for sym, tf, _, ba, bars in datasets:
        mr_sigs = collect_mr_signals(bars)
        r = replay_trades(mr_sigs, ba)
        wr = r["wins"]/r["trades"] if r["trades"] else 0
        print(f"    {sym:<16} {tf:>4} {r['trades']:>7} {wr:>6.1%} "
              f"{r['pnl']:>+12,.2f}")

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.0f}s\n")


if __name__ == "__main__":
    main()
