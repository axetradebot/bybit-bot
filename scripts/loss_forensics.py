"""
Loss forensics: what causes losses and how to reduce them.

Analyses:
1. MFE/MAE -- how far did trades go in your favor/against before exit?
2. SL tightness -- are stops too tight? Do losers almost win first?
3. Indicator values at entry -- winners vs losers
4. Bars held -- are quick exits worse?
5. Direction breakdown -- long vs short edge

Usage
-----
    python scripts/loss_forensics.py
    python scripts/loss_forensics.py --symbols SOLUSDT 1000PEPEUSDT
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

CURRENT_SYMBOLS = ["SOLUSDT", "AVAXUSDT", "WIFUSDT", "DOTUSDT", "1000PEPEUSDT"]
TIMEFRAMES = ["15m", "4h"]
EQUITY = 1_000.0
RISK_PCT = 0.02
LEVERAGE = 10

NEW_ALTCOINS = [
    "DOGEUSDT", "XRPUSDT", "LINKUSDT", "NEARUSDT",
    "APTUSDT", "SUIUSDT", "LTCUSDT", "AAVEUSDT",
    "OPUSDT", "INJUSDT", "FETUSDT", "RUNEUSDT",
    "ENAUSDT", "ONDOUSDT", "JUPUSDT", "WUSDT",
    "TIAUSDT", "SEIUSDT", "STXUSDT", "RENDERUSDT",
]

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
        print(f"  [SKIP] {symbol} not found on Bybit", flush=True)
        return pd.DataFrame()

    since_ts = int(datetime.fromisoformat(since).replace(tzinfo=timezone.utc).timestamp() * 1000)
    until_ts = int(datetime.fromisoformat(until).replace(tzinfo=timezone.utc).timestamp() * 1000)
    all_candles: list[list] = []
    current = since_ts
    while current < until_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(bybit_symbol, "5m", since=current, limit=1000)
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


def run_backtest(symbol, tf, trading_bars, context_df, from_date, to_date):
    strat = SniperStrategy()
    adapter = StrategyAdapter(strat, None, symbol, from_date, to_date, context_df=context_df)
    sim = Simulator(
        strategy=adapter, symbol=symbol,
        leverage=LEVERAGE, risk_pct=RISK_PCT,
        equity=EQUITY, fixed_risk=True,
    )
    return sim.run(None, from_date, to_date, bars=trading_bars)


def pct(values):
    """Percentile helper."""
    if not values:
        return 0, 0, 0, 0
    arr = np.array(values)
    return np.percentile(arr, 25), np.median(arr), np.percentile(arr, 75), np.mean(arr)


# ====================================================================
# PART 1: Loss Forensics
# ====================================================================

def loss_forensics(all_trades: list[ClosedTrade]):
    print(f"\n{'='*90}")
    print(f"  LOSS FORENSICS")
    print(f"{'='*90}")

    winners = [t for t in all_trades if t.win_loss]
    losers = [t for t in all_trades if not t.win_loss]
    total = len(all_trades)

    print(f"\n  Total trades: {total}  |  Winners: {len(winners)}  |  Losers: {len(losers)}")
    print(f"  Win rate: {len(winners)/total:.1%}")
    print(f"  Total PnL: {sum(t.pnl_usd for t in all_trades):+,.2f}")

    # --- MFE Analysis ---
    print(f"\n  {'='*80}")
    print(f"  MAX FAVORABLE EXCURSION (MFE) -- how far did trades go in your favor?")
    print(f"  {'='*80}")

    w_mfe = [t.mfe_pct * 100 for t in winners]
    l_mfe = [t.mfe_pct * 100 for t in losers]

    print(f"\n  {'':>12} {'P25':>8} {'Median':>8} {'P75':>8} {'Mean':>8}")
    p25, med, p75, mean = pct(w_mfe)
    print(f"  {'Winners':<12} {p25:>7.2f}% {med:>7.2f}% {p75:>7.2f}% {mean:>7.2f}%")
    p25, med, p75, mean = pct(l_mfe)
    print(f"  {'Losers':<12} {p25:>7.2f}% {med:>7.2f}% {p75:>7.2f}% {mean:>7.2f}%")

    # How many losers went 2%+ in their favor before reversing?
    near_miss_2 = sum(1 for t in losers if t.mfe_pct >= 0.02)
    near_miss_5 = sum(1 for t in losers if t.mfe_pct >= 0.05)
    near_miss_10 = sum(1 for t in losers if t.mfe_pct >= 0.10)
    print(f"\n  Losers that went in your favor BEFORE reversing to SL:")
    print(f"    >= 2% in favor then lost: {near_miss_2:>5} ({near_miss_2/len(losers):.1%} of losers)")
    print(f"    >= 5% in favor then lost: {near_miss_5:>5} ({near_miss_5/len(losers):.1%} of losers)")
    print(f"    >=10% in favor then lost: {near_miss_10:>5} ({near_miss_10/len(losers):.1%} of losers)")

    near_miss_pnl = sum(t.pnl_usd for t in losers if t.mfe_pct >= 0.05)
    print(f"    PnL from losers with >=5% MFE: {near_miss_pnl:+,.2f}")

    # --- MAE Analysis ---
    print(f"\n  {'='*80}")
    print(f"  MAX ADVERSE EXCURSION (MAE) -- how far did trades go against you?")
    print(f"  {'='*80}")

    w_mae = [t.mae_pct * 100 for t in winners]
    l_mae = [t.mae_pct * 100 for t in losers]

    print(f"\n  {'':>12} {'P25':>8} {'Median':>8} {'P75':>8} {'Mean':>8}")
    p25, med, p75, mean = pct(w_mae)
    print(f"  {'Winners':<12} {p25:>7.2f}% {med:>7.2f}% {p75:>7.2f}% {mean:>7.2f}%")
    p25, med, p75, mean = pct(l_mae)
    print(f"  {'Losers':<12} {p25:>7.2f}% {med:>7.2f}% {p75:>7.2f}% {mean:>7.2f}%")

    # --- SL Tightness ---
    print(f"\n  {'='*80}")
    print(f"  SL TIGHTNESS -- is the stop too tight?")
    print(f"  {'='*80}")

    sl_distances_pct = []
    for t in all_trades:
        if t.entry_price > 0:
            sl_dist = abs(t.stop_loss - t.entry_price) / t.entry_price * 100
            sl_distances_pct.append(sl_dist)

    p25, med, p75, mean = pct(sl_distances_pct)
    print(f"\n  SL distance from entry: P25={p25:.2f}%  Med={med:.2f}%  "
          f"P75={p75:.2f}%  Mean={mean:.2f}%")

    instant_losses = sum(1 for t in losers if t.mfe_pct < 0.002)
    print(f"  Losers that never went >0.2% in favor (instant SL): "
          f"{instant_losses} ({instant_losses/len(losers):.1%})")

    # --- Indicator analysis: winners vs losers ---
    print(f"\n  {'='*80}")
    print(f"  ENTRY INDICATOR VALUES: Winners vs Losers")
    print(f"  {'='*80}")

    indicator_keys = ["rsi_14", "atr_pct_rank", "bb_squeeze", "mfi_14"]
    print(f"\n  {'Indicator':<16} {'Winners Med':>12} {'Losers Med':>12} {'Delta':>8}")

    for key in indicator_keys:
        w_vals = [t.indicators_snapshot.get(key) for t in winners
                  if t.indicators_snapshot.get(key) is not None]
        l_vals = [t.indicators_snapshot.get(key) for t in losers
                  if t.indicators_snapshot.get(key) is not None]
        if w_vals and l_vals:
            w_med = np.median(w_vals)
            l_med = np.median(l_vals)
            print(f"  {key:<16} {w_med:>12.4f} {l_med:>12.4f} {w_med-l_med:>+8.4f}")

    # --- Direction breakdown ---
    print(f"\n  {'='*80}")
    print(f"  DIRECTION BREAKDOWN")
    print(f"  {'='*80}")

    for direction in ["long", "short"]:
        dt = [t for t in all_trades if t.direction == direction]
        if not dt:
            continue
        dw = sum(1 for t in dt if t.win_loss)
        dpnl = sum(t.pnl_usd for t in dt)
        dmfe = np.mean([t.mfe_pct * 100 for t in dt])
        print(f"\n  {direction.upper()}: {len(dt)} trades  WR {dw/len(dt):.1%}  "
              f"PnL {dpnl:+,.2f}  Avg MFE {dmfe:.2f}%")

    # --- Per-symbol loss analysis ---
    print(f"\n  {'='*80}")
    print(f"  PER-SYMBOL LOSS BREAKDOWN")
    print(f"  {'='*80}")

    by_sym: dict[str, list[ClosedTrade]] = defaultdict(list)
    for t in all_trades:
        by_sym[t.symbol].append(t)

    print(f"\n  {'Symbol':<16} {'Trades':>6} {'WR':>6} {'PnL':>10} "
          f"{'Avg MFE':>8} {'Avg MAE':>8} {'InstantSL':>9}")
    for sym in sorted(by_sym):
        st = by_sym[sym]
        sw = sum(1 for t in st if t.win_loss)
        spnl = sum(t.pnl_usd for t in st)
        smfe = np.mean([t.mfe_pct * 100 for t in st])
        smae = np.mean([t.mae_pct * 100 for t in st])
        sl_losers = [t for t in st if not t.win_loss]
        instant = sum(1 for t in sl_losers if t.mfe_pct < 0.002) if sl_losers else 0
        inst_pct = instant / len(sl_losers) if sl_losers else 0
        print(f"  {sym:<16} {len(st):>6} {sw/len(st):>6.1%} {spnl:>+10,.2f} "
              f"{smfe:>7.2f}% {smae:>7.2f}% {instant:>5} ({inst_pct:.0%})")


# ====================================================================
# PART 2: Altcoin scan
# ====================================================================

def altcoin_scan(from_date, to_date, timeframes):
    print(f"\n{'='*90}")
    print(f"  NEW ALTCOIN SCAN -- {len(NEW_ALTCOINS)} candidates")
    print(f"  Testing: {', '.join(timeframes)}")
    print(f"{'='*90}\n")

    results = []
    for idx, sym in enumerate(NEW_ALTCOINS):
        cache_dir = project_root / "data_cache"
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / f"{sym}_{from_date}_{to_date}_5m.parquet"

        if cache_path.exists():
            print(f"[{idx+1}/{len(NEW_ALTCOINS)}] {sym} -- cache ...", end=" ", flush=True)
            df = pd.read_parquet(cache_path)
        else:
            print(f"[{idx+1}/{len(NEW_ALTCOINS)}] {sym} -- downloading ...", end=" ", flush=True)
            df = fetch_5m_candles(sym, from_date, to_date)
            if not df.empty:
                df.to_parquet(cache_path)

        if df.empty or len(df) < 500:
            print(f"SKIP ({len(df)} candles)", flush=True)
            continue
        print(f"{len(df):,} candles", flush=True)

        for tf in timeframes:
            trading_bars = build_bars_for_tf(df, tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            context_df = build_bars_for_tf(df, ctx_tf) if ctx_tf != tf else trading_bars

            if trading_bars is None or trading_bars.empty:
                continue

            trades = run_backtest(sym, tf, trading_bars, context_df, from_date, to_date)
            if not trades:
                print(f"  {tf:>3}: 0 trades", flush=True)
                continue

            usd = compute_pnl_dollar_summary(trades, EQUITY)
            wins = sum(1 for t in trades if t.win_loss)
            pnl_pcts = [t.pnl_pct for t in trades]
            d0 = datetime.fromisoformat(from_date)
            d1 = datetime.fromisoformat(to_date)
            days = max((d1 - d0).days, 1)
            trades_per_year = len(trades) / days * 365
            sharpe = compute_sharpe(pnl_pcts, trades_per_year)

            tag = "+++" if usd["total_pnl_usd"] > 0 else "---"
            print(f"  {tf:>3}: {len(trades):>4} trades  WR {wins/len(trades):5.1%}  "
                  f"PnL {usd['total_pnl_usd']:>+10,.2f}  "
                  f"Sharpe {sharpe:.2f}  {tag}", flush=True)

            results.append({
                "symbol": sym, "tf": tf, "trades": len(trades),
                "wr": wins / len(trades), "pnl": usd["total_pnl_usd"],
                "sharpe": sharpe, "pf": usd["profit_factor"],
                "max_dd": usd["max_dd_vs_peak_pct"],
                "payoff": usd["payoff_ratio"],
            })

    # Summary
    print(f"\n{'='*90}")
    print(f"  ALTCOIN SCAN RESULTS (sorted by PnL)")
    print(f"{'='*90}")
    print(f"  {'Symbol':<14} {'TF':>4} {'Trades':>6} {'WR':>6} {'PnL':>10} "
          f"{'Sharpe':>7} {'PF':>6} {'MaxDD':>7} {'Payoff':>7}")
    print(f"  {'-'*75}")

    for r in sorted(results, key=lambda x: x["pnl"], reverse=True):
        pf_s = f"{r['pf']:.2f}" if not np.isinf(r["pf"]) else "inf"
        print(f"  {r['symbol']:<14} {r['tf']:>4} {r['trades']:>6} "
              f"{r['wr']:>6.1%} {r['pnl']:>+10,.2f} "
              f"{r['sharpe']:>7.2f} {pf_s:>6} {r['max_dd']:>6.1%} "
              f"{r['payoff']:>6.2f}x")

    good = [r for r in results if r["pnl"] > 100 and r["pf"] > 1.2 and r["sharpe"] > 0.3]
    if good:
        print(f"\n  RECOMMENDED ADDITIONS (PnL>$100, PF>1.2, Sharpe>0.3):")
        for r in sorted(good, key=lambda x: x["pnl"], reverse=True):
            print(f"    {r['symbol']:<14} {r['tf']:>4}  PnL {r['pnl']:>+10,.2f}  "
                  f"Sharpe {r['sharpe']:.2f}")

    return results


# ====================================================================
# PART 3: 1h timeframe test on current symbols
# ====================================================================

def test_1h_timeframe(candle_cache, symbols, from_date, to_date):
    print(f"\n{'='*90}")
    print(f"  1H TIMEFRAME TEST -- on current symbols")
    print(f"{'='*90}")

    print(f"\n  {'Symbol':<16} {'TF':>4} {'Trades':>6} {'WR':>6} {'PnL':>10} "
          f"{'Sharpe':>7} {'MaxDD':>7}")

    for sym in symbols:
        df_5m = candle_cache.get(sym)
        if df_5m is None or df_5m.empty:
            continue

        for tf in ["15m", "1h", "4h"]:
            trading_bars = build_bars_for_tf(df_5m, tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            context_df = build_bars_for_tf(df_5m, ctx_tf) if ctx_tf != tf else trading_bars

            trades = run_backtest(sym, tf, trading_bars, context_df, from_date, to_date)
            if not trades:
                print(f"  {sym:<16} {tf:>4}      0 trades")
                continue

            usd = compute_pnl_dollar_summary(trades, EQUITY)
            wins = sum(1 for t in trades if t.win_loss)
            pnl_pcts = [t.pnl_pct for t in trades]
            d0, d1 = datetime.fromisoformat(from_date), datetime.fromisoformat(to_date)
            tpy = len(trades) / max((d1 - d0).days, 1) * 365
            sharpe = compute_sharpe(pnl_pcts, tpy)

            marker = " <-- NEW" if tf == "1h" else ""
            print(f"  {sym:<16} {tf:>4} {len(trades):>6} {wins/len(trades):>6.1%} "
                  f"{usd['total_pnl_usd']:>+10,.2f} {sharpe:>7.2f} "
                  f"{usd['max_dd_vs_peak_pct']:>6.1%}{marker}")


def main():
    p = argparse.ArgumentParser(description="Loss forensics + altcoin scan")
    p.add_argument("--from", dest="from_date", default="2023-04-01")
    p.add_argument("--to", dest="to_date",
                   default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--symbols", nargs="+", default=CURRENT_SYMBOLS)
    p.add_argument("--skip-altcoins", action="store_true")
    args = p.parse_args()

    print(f"\n{'='*90}")
    print(f"  LOSS FORENSICS + EXPANSION ANALYSIS")
    print(f"  {args.from_date} -> {args.to_date}")
    print(f"{'='*90}\n")

    t_start = time.time()

    # Load current symbols
    candle_cache = load_candles(args.symbols, args.from_date, args.to_date)

    # Collect all trades
    all_trades: list[ClosedTrade] = []
    for sym in args.symbols:
        df_5m = candle_cache[sym]
        if df_5m.empty:
            continue
        for tf in TIMEFRAMES:
            trading_bars = build_bars_for_tf(df_5m, tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            context_df = build_bars_for_tf(df_5m, ctx_tf) if ctx_tf != tf else trading_bars
            trades = run_backtest(sym, tf, trading_bars, context_df,
                                  args.from_date, args.to_date)
            all_trades.extend(trades)

    # Part 1: Loss forensics
    loss_forensics(all_trades)

    # Part 2: 1h timeframe
    test_1h_timeframe(candle_cache, args.symbols, args.from_date, args.to_date)

    # Part 3: Altcoin scan
    if not args.skip_altcoins:
        altcoin_scan(args.from_date, args.to_date, ["15m", "1h", "4h"])

    print(f"\n  Total elapsed: {time.time() - t_start:.0f}s\n")


if __name__ == "__main__":
    main()
