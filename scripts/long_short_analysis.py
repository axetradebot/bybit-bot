"""
Long vs Short analysis by year, plus stricter long filter testing.

Tests whether the short-side edge is consistent across all years and
whether tightening long-only filters improves overall PnL.

Usage
-----
    python scripts/long_short_analysis.py
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.backtest.simulator import Simulator, ClosedTrade
from src.backtest.run_backtest import compute_pnl_dollar_summary, compute_sharpe
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies import StrategyAdapter
from src.strategies.strategy_sniper import SniperStrategy

log = structlog.get_logger()

ALL_SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT", "DOTUSDT", "1000PEPEUSDT",
    "DOGEUSDT", "OPUSDT", "NEARUSDT", "SUIUSDT",
]
TIMEFRAMES = ["15m", "4h"]
EQUITY = 1_000.0
RISK_PCT = 0.02
LEVERAGE = 10

Simulator._load_funding = lambda self, engine: {}


def fetch_5m_candles(symbol: str, since: str, until: str) -> pd.DataFrame:
    import ccxt
    exchange = ccxt.bybit({"enableRateLimit": True})
    exchange.load_markets()
    from src.live.order_manager import _to_ccxt_symbol
    bybit_symbol = _to_ccxt_symbol(symbol)
    if bybit_symbol not in exchange.markets:
        bybit_symbol = symbol.replace("USDT", "/USDT:USDT")
    if bybit_symbol not in exchange.markets:
        bybit_symbol = symbol.replace("USDT", "/USDT")
    if bybit_symbol not in exchange.markets:
        return pd.DataFrame()
    since_ts = int(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp() * 1000)
    until_ts = int(datetime.fromisoformat(until).replace(tzinfo=timezone.utc).timestamp() * 1000)
    all_candles: list[list] = []
    current = since_ts
    while current < until_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(bybit_symbol, "5m", since=current, limit=1000)
        except Exception as e:
            time.sleep(2)
            continue
        if not ohlcv:
            break
        for c in ohlcv:
            if c[0] >= until_ts:
                break
            all_candles.append(c)
        last_ts = ohlcv[-1][0]
        if last_ts <= current:
            break
        current = last_ts + 1
    if not all_candles:
        return pd.DataFrame()
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["symbol"] = symbol
    df["buy_volume"] = df["volume"] * 0.5
    df["sell_volume"] = df["volume"] * 0.5
    df["volume_delta"] = 0.0
    df["quote_volume"] = df["volume"] * df["close"]
    df["trade_count"] = 0
    df["mark_price"] = df["close"]
    df["funding_rate"] = 0.0
    return df


def load_candles(symbols, from_date, to_date):
    cache_dir = project_root / "data_cache"
    cache_dir.mkdir(exist_ok=True)
    cache = {}
    for i, sym in enumerate(symbols):
        cache_path = cache_dir / f"{sym}_{from_date}_{to_date}_5m.parquet"
        if cache_path.exists():
            print(f"[{i+1}/{len(symbols)}] {sym} -- cache ...", end=" ", flush=True)
            df = pd.read_parquet(cache_path)
        else:
            print(f"[{i+1}/{len(symbols)}] {sym} -- downloading ...", end=" ", flush=True)
            df = fetch_5m_candles(sym, from_date, to_date)
            if not df.empty:
                df.to_parquet(cache_path)
        print(f"{len(df):,} candles", flush=True)
        cache[sym] = df
    return cache


def run_backtest(symbol, tf, trading_bars, context_df, from_date, to_date,
                 **sniper_kwargs):
    strat = SniperStrategy(**sniper_kwargs)
    adapter = StrategyAdapter(strat, None, symbol, from_date, to_date, context_df=context_df)
    sim = Simulator(strategy=adapter, symbol=symbol,
                    leverage=LEVERAGE, risk_pct=RISK_PCT,
                    equity=EQUITY, fixed_risk=True)
    return sim.run(None, from_date, to_date, bars=trading_bars)


def collect_all_trades(candle_cache, symbols, timeframes, from_date, to_date,
                       **sniper_kwargs):
    all_trades = []
    for sym in symbols:
        df_5m = candle_cache.get(sym)
        if df_5m is None or df_5m.empty:
            continue
        for tf in timeframes:
            trading_bars = build_bars_for_tf(df_5m, tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            context_df = build_bars_for_tf(df_5m, ctx_tf) if ctx_tf != tf else trading_bars
            trades = run_backtest(sym, tf, trading_bars, context_df,
                                  from_date, to_date, **sniper_kwargs)
            all_trades.extend(trades)
    return all_trades


# ====================================================================
# PART 1: Long vs Short by year, per symbol
# ====================================================================

def long_short_yearly(all_trades: list[ClosedTrade]):
    print(f"\n{'='*90}")
    print(f"  LONG vs SHORT BY YEAR -- Is the short edge consistent?")
    print(f"{'='*90}")

    # Aggregate by year and direction
    by_year_dir: dict[int, dict[str, list[ClosedTrade]]] = defaultdict(lambda: defaultdict(list))
    for t in all_trades:
        by_year_dir[t.entry_time.year][t.direction].append(t)

    print(f"\n  {'Year':<6} {'Dir':<7} {'Trades':>6} {'WR':>6} {'PnL':>12} "
          f"{'AvgWin':>8} {'AvgLoss':>8}")
    print(f"  {'-'*65}")

    for yr in sorted(by_year_dir):
        for direction in ["long", "short"]:
            dt = by_year_dir[yr][direction]
            if not dt:
                continue
            w = sum(1 for t in dt if t.win_loss)
            pnl = sum(t.pnl_usd for t in dt)
            aw = np.mean([t.pnl_usd for t in dt if t.win_loss]) if w else 0
            al = np.mean([t.pnl_usd for t in dt if not t.win_loss]) if (len(dt) - w) else 0
            print(f"  {yr:<6} {direction:<7} {len(dt):>6} {w/len(dt):>6.1%} "
                  f"{pnl:>+12,.2f} {aw:>+8.2f} {al:>+8.2f}")
        # Year total
        all_yr = by_year_dir[yr]["long"] + by_year_dir[yr]["short"]
        pnl_yr = sum(t.pnl_usd for t in all_yr)
        print(f"  {yr:<6} {'TOTAL':<7} {len(all_yr):>6} "
              f"{sum(1 for t in all_yr if t.win_loss)/len(all_yr):>6.1%} "
              f"{pnl_yr:>+12,.2f}")
        print()

    # Per-symbol long vs short
    print(f"\n  {'='*80}")
    print(f"  PER-SYMBOL: Long vs Short (all years)")
    print(f"  {'='*80}")

    by_sym: dict[str, list[ClosedTrade]] = defaultdict(list)
    for t in all_trades:
        by_sym[t.symbol].append(t)

    print(f"\n  {'Symbol':<16} {'Long Trades':>11} {'Long WR':>8} {'Long PnL':>10} "
          f"{'Short Trades':>12} {'Short WR':>9} {'Short PnL':>10}")
    print(f"  {'-'*85}")

    for sym in sorted(by_sym):
        longs = [t for t in by_sym[sym] if t.direction == "long"]
        shorts = [t for t in by_sym[sym] if t.direction == "short"]
        lw = sum(1 for t in longs if t.win_loss) if longs else 0
        sw = sum(1 for t in shorts if t.win_loss) if shorts else 0
        lpnl = sum(t.pnl_usd for t in longs)
        spnl = sum(t.pnl_usd for t in shorts)
        lwr = lw / len(longs) if longs else 0
        swr = sw / len(shorts) if shorts else 0
        print(f"  {sym:<16} {len(longs):>11} {lwr:>8.1%} {lpnl:>+10,.2f} "
              f"{len(shorts):>12} {swr:>9.1%} {spnl:>+10,.2f}")


# ====================================================================
# PART 2: Stricter long filters
# ====================================================================

def test_long_filters(candle_cache, symbols, timeframes, from_date, to_date):
    print(f"\n{'='*90}")
    print(f"  STRICTER LONG FILTERS -- Testing tighter RSI & EMA spread for longs")
    print(f"{'='*90}")

    configs = [
        {"label": "Current (baseline)",
         "params": {}},
        {"label": "Tighter long RSI (35-55)",
         "params": {"rsi_long_lo": 35, "rsi_long_hi": 55}},
        {"label": "Tighter long RSI (38-50)",
         "params": {"rsi_long_lo": 38, "rsi_long_hi": 50}},
        {"label": "Higher EMA spread for longs (0.005)",
         "params": {"ema_spread_min": 0.005}},
        {"label": "Higher ATR rank floor (0.40)",
         "params": {"atr_rank_floor": 0.40}},
        {"label": "Combo: RSI 35-55 + EMA 0.005",
         "params": {"rsi_long_lo": 35, "rsi_long_hi": 55, "ema_spread_min": 0.005}},
        {"label": "Combo: RSI 35-55 + ATR 0.40",
         "params": {"rsi_long_lo": 35, "rsi_long_hi": 55, "atr_rank_floor": 0.40}},
        {"label": "Nuclear: RSI 38-50 + EMA 0.005 + ATR 0.40",
         "params": {"rsi_long_lo": 38, "rsi_long_hi": 50, "ema_spread_min": 0.005,
                    "atr_rank_floor": 0.40}},
    ]

    print(f"\n  Note: Tighter params also affect SHORTS (EMA spread, ATR rank).")
    print(f"  RSI params are direction-specific (only change long RSI range).\n")

    hdr = (f"  {'Config':<42} {'Total':>6} {'Long':>5} {'Short':>5} "
           f"{'L-PnL':>10} {'S-PnL':>10} {'Total PnL':>10} {'Delta':>8}")
    print(hdr)
    print(f"  {'-'*100}")

    baseline_pnl = None

    for cfg in configs:
        trades = collect_all_trades(candle_cache, symbols, timeframes,
                                    from_date, to_date, **cfg["params"])

        longs = [t for t in trades if t.direction == "long"]
        shorts = [t for t in trades if t.direction == "short"]
        l_pnl = sum(t.pnl_usd for t in longs)
        s_pnl = sum(t.pnl_usd for t in shorts)
        total_pnl = l_pnl + s_pnl

        if baseline_pnl is None:
            baseline_pnl = total_pnl

        delta = total_pnl - baseline_pnl
        print(f"  {cfg['label']:<42} {len(trades):>6} {len(longs):>5} {len(shorts):>5} "
              f"{l_pnl:>+10,.2f} {s_pnl:>+10,.2f} {total_pnl:>+10,.2f} "
              f"{delta:>+8,.0f}")

    # Short-only mode comparison
    print(f"\n  {'='*80}")
    print(f"  SHORT-ONLY MODE")
    print(f"  {'='*80}")

    all_trades = collect_all_trades(candle_cache, symbols, timeframes,
                                    from_date, to_date)
    shorts_only = [t for t in all_trades if t.direction == "short"]
    longs_only = [t for t in all_trades if t.direction == "long"]

    print(f"\n  Longs total:  {len(longs_only):>5} trades  PnL {sum(t.pnl_usd for t in longs_only):>+10,.2f}")
    print(f"  Shorts total: {len(shorts_only):>5} trades  PnL {sum(t.pnl_usd for t in shorts_only):>+10,.2f}")
    print(f"  Combined:     {len(all_trades):>5} trades  PnL {sum(t.pnl_usd for t in all_trades):>+10,.2f}")
    print(f"  Short-only:   {len(shorts_only):>5} trades  PnL {sum(t.pnl_usd for t in shorts_only):>+10,.2f}")

    # Short-only by year
    print(f"\n  Short-only by year:")
    by_yr: dict[int, list[ClosedTrade]] = defaultdict(list)
    for t in shorts_only:
        by_yr[t.entry_time.year].append(t)
    for yr in sorted(by_yr):
        st = by_yr[yr]
        w = sum(1 for t in st if t.win_loss)
        pnl = sum(t.pnl_usd for t in st)
        print(f"    {yr}: {len(st):>5} trades  WR {w/len(st):.1%}  PnL {pnl:>+10,.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_date", default="2023-04-01")
    p.add_argument("--to", dest="to_date",
                   default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--symbols", nargs="+", default=ALL_SYMBOLS)
    args = p.parse_args()

    print(f"\n{'='*90}")
    print(f"  LONG vs SHORT ANALYSIS + FILTER OPTIMIZATION")
    print(f"  {args.from_date} -> {args.to_date}")
    print(f"  Symbols: {', '.join(args.symbols)}")
    print(f"{'='*90}\n")

    t_start = time.time()
    candle_cache = load_candles(args.symbols, args.from_date, args.to_date)

    # Part 1: Baseline long vs short yearly
    baseline_trades = collect_all_trades(candle_cache, args.symbols, TIMEFRAMES,
                                         args.from_date, args.to_date)
    long_short_yearly(baseline_trades)

    # Part 2: Test stricter long filters
    test_long_filters(candle_cache, args.symbols, TIMEFRAMES,
                      args.from_date, args.to_date)

    print(f"\n  Total elapsed: {time.time() - t_start:.0f}s\n")


if __name__ == "__main__":
    main()
