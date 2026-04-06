"""
SIMULATION ONLY — does not change the live bot or SniperStrategy.

Collects baseline Sniper signals using the **live-aligned** engine in
``src/backtest/live_aligned_portfolio.py`` (aggressive limit entry + TTL,
taker entry fee, maker on TP / taker on SL-trail-timeout, slippage on
stops, pro-rata 8h funding from bar ``funding_8h`` when present else flat
rate), then replays the shared-equity portfolio with post-hoc filters.

Run:
    python scripts/anti_chop_sweep_sim.py

Ideas tested (toggle via scenario predicates):
  From prior plan:
    - ADX floor (15/20/25/30)
    - +DI / -DI aligned with direction
    - volume_ratio floors (0.8 / 1.0 / 1.2 / 1.5)
    - VWAP side (long above, short below)
    - skip BB squeeze; BB width vs price; position vs BB mid (upper/lower half)
    - OBV slope sign vs direction
    - candle_body_ratio floor
    - MACD hist on 5m; MACD hist on 5m + 15m context aligned
  Extra anti-chop ideas:
    - Stricter atr_pct_rank (0.35 / 0.40 / 0.50)
    - Bar range vs ATR (signal bar must “move” enough)
    - CMF sign vs direction
    - ROC sign vs direction
    - MACD *fast* hist aligned with direction
    - MFI not stuck mid-band (chop proxy)
    - StochRSI K not at extreme (reduce fake reversal entries)
    - Williams %R band
    - Combo stacks (ADX+DI+volume, etc.)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable

os.environ["PYTHONUNBUFFERED"] = "1"

import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.base import _sf
from src.backtest.live_aligned_portfolio import (
    LivePortfolioTrade,
    collect_trades_live_aligned,
    run_portfolio_live_aligned,
)

PrecomputedTrade = LivePortfolioTrade

LEVERAGE = 20
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


def _snap(bar: pd.Series, ctx: pd.Series) -> dict:
    """Indicator snapshot at signal time (5m bar + context row)."""
    return {
        "adx_14": _sf(bar.get("adx_14")),
        "plus_di": _sf(bar.get("plus_di")),
        "minus_di": _sf(bar.get("minus_di")),
        "bb_width": _sf(bar.get("bb_width")),
        "bb_squeeze": bool(bar.get("bb_squeeze", False)),
        "bb_mid": _sf(bar.get("bb_mid")),
        "volume_ratio": _sf(bar.get("volume_ratio")),
        "vwap": _sf(bar.get("vwap")),
        "close": _sf(bar.get("close")),
        "high": _sf(bar.get("high")),
        "low": _sf(bar.get("low")),
        "obv_slope": _sf(bar.get("obv_slope")),
        "candle_body_ratio": _sf(bar.get("candle_body_ratio")),
        "macd_hist": _sf(bar.get("macd_hist")),
        "macd_fast_hist": _sf(bar.get("macd_fast_hist")),
        "macd_hist_ctx": _sf(ctx.get("macd_hist")),
        "cmf_20": _sf(bar.get("cmf_20")),
        "roc_10": _sf(bar.get("roc_10")),
        "willr_14": _sf(bar.get("willr_14")),
        "mfi_14": _sf(bar.get("mfi_14")),
        "stochrsi_k": _sf(bar.get("stochrsi_k")),
        "atr_pct_rank": _sf(bar.get("atr_pct_rank")),
        "atr_14": _sf(bar.get("atr_14")),
    }


def collect_trades_with_snapshots(
    trading_bars, context_df, symbol: str, tf: str,
) -> list[tuple[LivePortfolioTrade, dict]]:
    return collect_trades_live_aligned(
        trading_bars,
        context_df,
        symbol,
        tf,
        snap_fn=lambda b, c: _snap(b, c),
    )


def run_portfolio(all_trades: list[LivePortfolioTrade]) -> dict:
    return run_portfolio_live_aligned(
        all_trades,
        start_equity=START_EQUITY,
        risk_pct=RISK_PCT,
        max_concurrent=MAX_CONCURRENT,
    )


def di_aligned(s: dict, direction: str) -> bool:
    p, m = s["plus_di"], s["minus_di"]
    if p <= 0 and m <= 0:
        return True
    if direction == "long":
        return p > m
    return m > p


def macd_aligned(hist: float, direction: str) -> bool:
    if direction == "long":
        return hist > 0
    return hist < 0


def build_scenarios() -> list[tuple[str, Callable[[dict, str], bool]]]:
    """(label, passes(snap, direction))"""

    def adx_floor(x: float):
        def f(s: dict, d: str) -> bool:
            a = s["adx_14"]
            return a >= x if a > 0 else True

        return f

    def vol_floor(x: float):
        def f(s: dict, d: str) -> bool:
            v = s["volume_ratio"]
            return v >= x if v > 0 else True

        return f

    scenarios: list[tuple[str, Callable[[dict, str], bool]]] = [
        ("baseline (no extra filter)", lambda s, d: True),

        ("ADX >= 15", adx_floor(15)),
        ("ADX >= 20", adx_floor(20)),
        ("ADX >= 25", adx_floor(25)),
        ("ADX >= 30", adx_floor(30)),

        ("+DI/-DI aligned", lambda s, d: di_aligned(s, d)),
        ("ADX>=20 AND DI aligned", lambda s, d: (
            (s["adx_14"] >= 20 or s["adx_14"] <= 0) and di_aligned(s, d))),
        ("ADX>=25 AND DI aligned", lambda s, d: (
            (s["adx_14"] >= 25 or s["adx_14"] <= 0) and di_aligned(s, d))),

        ("volume_ratio >= 0.8", vol_floor(0.8)),
        ("volume_ratio >= 1.0", vol_floor(1.0)),
        ("volume_ratio >= 1.2", vol_floor(1.2)),
        ("volume_ratio >= 1.5", vol_floor(1.5)),

        ("VWAP side (long>vwap short<vwap)", lambda s, d: (
            s["vwap"] > 0 and (
                (d == "long" and s["close"] > s["vwap"]) or
                (d == "short" and s["close"] < s["vwap"])))),

        ("skip BB squeeze", lambda s, d: not s["bb_squeeze"]),

        ("bb_width / close > 1.2%", lambda s, d: (
            s["close"] > 0 and s["bb_width"] / s["close"] > 0.012)),

        ("close vs BB mid (upper/lower half)", lambda s, d: (
            s["bb_mid"] > 0 and (
                (d == "long" and s["close"] > s["bb_mid"]) or
                (d == "short" and s["close"] < s["bb_mid"])))),

        ("OBV slope aligned", lambda s, d: (
            (d == "long" and s["obv_slope"] > 0) or
            (d == "short" and s["obv_slope"] < 0))),

        ("candle_body_ratio >= 0.3", lambda s, d: s["candle_body_ratio"] >= 0.3),
        ("candle_body_ratio >= 0.4", lambda s, d: s["candle_body_ratio"] >= 0.4),
        ("candle_body_ratio >= 0.5", lambda s, d: s["candle_body_ratio"] >= 0.5),

        ("MACD hist 5m aligned", lambda s, d: macd_aligned(s["macd_hist"], d)),
        ("MACD hist 5m + 15m ctx aligned", lambda s, d: (
            macd_aligned(s["macd_hist"], d) and
            (s["macd_hist_ctx"] != 0) and macd_aligned(s["macd_hist_ctx"], d))),

        ("MACD fast hist aligned", lambda s, d: macd_aligned(
            s["macd_fast_hist"], d)),

        ("CMF aligned (long>0 short<0)", lambda s, d: (
            (d == "long" and s["cmf_20"] > 0) or
            (d == "short" and s["cmf_20"] < 0))),

        ("ROC aligned", lambda s, d: (
            (d == "long" and s["roc_10"] > 0) or
            (d == "short" and s["roc_10"] < 0))),

        ("atr_pct_rank >= 0.35", lambda s, d: s["atr_pct_rank"] >= 0.35),
        ("atr_pct_rank >= 0.40", lambda s, d: s["atr_pct_rank"] >= 0.40),
        ("atr_pct_rank >= 0.50", lambda s, d: s["atr_pct_rank"] >= 0.50),

        ("bar range >= 0.45 * ATR", lambda s, d: (
            s["atr_14"] > 0 and (s["high"] - s["low"]) >= 0.45 * s["atr_14"])),

        ("MFI bias (long>42 short<58)", lambda s, d: (
            (d == "long" and s["mfi_14"] > 42) or
            (d == "short" and s["mfi_14"] < 58))),

        ("StochRSI K not extreme (long>18 short<82)", lambda s, d: (
            (d == "long" and s["stochrsi_k"] > 18) or
            (d == "short" and s["stochrsi_k"] < 82))),

        ("Williams %R band (long>-55 short<-45)", lambda s, d: (
            (d == "long" and s["willr_14"] > -55) or
            (d == "short" and s["willr_14"] < -45))),

        ("combo: ADX20+DI+vol1.0+no squeeze+VWAP", lambda s, d: (
            (s["adx_14"] >= 20 or s["adx_14"] <= 0) and di_aligned(s, d)
            and (s["volume_ratio"] >= 1.0 or s["volume_ratio"] <= 0)
            and not s["bb_squeeze"]
            and s["vwap"] > 0 and (
                (d == "long" and s["close"] > s["vwap"]) or
                (d == "short" and s["close"] < s["vwap"])))),

        ("combo: ADX25+DI+body0.4+MACD5m", lambda s, d: (
            (s["adx_14"] >= 25 or s["adx_14"] <= 0) and di_aligned(s, d)
            and s["candle_body_ratio"] >= 0.4
            and macd_aligned(s["macd_hist"], d))),
    ]
    return scenarios


def load_candles(symbol, cache_dir):
    p = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return pd.DataFrame()


def main():
    cache_dir = project_root / "data_cache"
    scenarios = build_scenarios()

    print("=" * 100, flush=True)
    print("ANTI-CHOP FILTER SWEEP (live-aligned sim; bot code unchanged)", flush=True)
    print(f"Shared equity ${START_EQUITY:,.0f}  |  risk {RISK_PCT:.0%}  |  "
          f"max concurrent {MAX_CONCURRENT}  |  leverage {LEVERAGE}x", flush=True)
    print("Engine: limit entry +/-0.01% + 3-bar TTL, taker entry, maker TP / "
          "taker SL-trail-timeout, stop slippage, 8h funding", flush=True)
    print("=" * 100, flush=True)

    print("\n-- Collect baseline trades + entry snapshots --", flush=True)
    rows: list[tuple[PrecomputedTrade, dict]] = []
    for sym in SYMBOLS:
        df_5m = load_candles(sym, cache_dir)
        if df_5m.empty:
            print(f"  skip {sym} (no cache)", flush=True)
            continue
        for tf in TIMEFRAMES:
            t0 = time.time()
            ctx_tf = CONTEXT_TF.get(tf, "4h")
            ctx_df = build_bars_for_tf(df_5m, ctx_tf)
            bars = build_bars_for_tf(df_5m, tf)
            chunk = collect_trades_with_snapshots(bars, ctx_df, sym, tf)
            rows.extend(chunk)
            print(f"  {tf} {sym}: {len(chunk)} trades ({time.time()-t0:.1f}s)",
                  flush=True)

    rows.sort(key=lambda x: x[0].entry_time)
    n_base = len(rows)
    print(f"\n  Total baseline trades: {n_base}", flush=True)

    print("\n-- Sweeping filters (portfolio replay) --", flush=True)
    hdr = (f"{'Scenario':<48s}  {'Kept':>6s}  {'Trades':>7s}  {'WR':>6s}  "
           f"{'Final$':>10s}  {'PnL':>11s}  {'Ret%':>8s}  {'MaxDD':>7s}  "
           f"{'MinEq':>9s}")
    print(hdr, flush=True)
    print("-" * 100, flush=True)

    results = []
    for label, pred in scenarios:
        filt = [t for t, snap in rows if pred(snap, t.direction)]
        r = run_portfolio(filt)
        kept = len(filt)
        ret = r["pnl"] / START_EQUITY * 100
        results.append((label, kept, r))
        print(f"{label:<48s}  {kept:>6d}  {r['trades']:>7d}  {r['wr']:>5.1f}%  "
              f"${r['final_eq']:>8,.0f}  ${r['pnl']:>+9,.0f}  {ret:>+7.1f}%  "
              f"{r['max_dd_pct']:>6.1%}  ${r['min_equity']:>7,.0f}", flush=True)

    best = max(results, key=lambda x: x[2]["final_eq"])
    safest = min(results, key=lambda x: x[2]["max_dd_pct"])
    print("\n" + "=" * 100, flush=True)
    print(f"Highest final equity: {best[0]}  ->  ${best[2]['final_eq']:,.0f}  "
          f"(max DD {best[2]['max_dd_pct']:.1%})", flush=True)
    print(f"Lowest max drawdown:  {safest[0]}  ->  DD {safest[2]['max_dd_pct']:.1%}  "
          f"final ${safest[2]['final_eq']:,.0f}", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
