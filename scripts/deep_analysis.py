"""
Deep strategy analysis: symbol breakdown, partial TP, max concurrent positions.

1. Per-symbol deep dive  -- WHY does each symbol win/lose? (year, direction, session)
2. Partial take-profit   -- close 50% at 3x ATR, let rest ride to 6x ATR
3. Max positions cap     -- limit concurrent open trades across all symbols

Usage
-----
    python scripts/deep_analysis.py
    python scripts/deep_analysis.py --symbols SOLUSDT ARBUSDT
"""

from __future__ import annotations

import argparse
import copy
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

from src.backtest.simulator import (
    Simulator, ClosedTrade, _OpenPosition, _PendingOrder,
    TAKER_FEE, MAKER_FEE, COOLDOWN_BARS,
    classify_volatility, classify_funding, classify_time_of_day,
)
from src.backtest.run_backtest import compute_pnl_dollar_summary, compute_sharpe
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies import StrategyAdapter
from src.strategies.strategy_sniper import SniperStrategy

log = structlog.get_logger()

ALL_SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT", "DOTUSDT",
    "1000PEPEUSDT",
]
TIMEFRAMES = ["15m", "4h"]
EQUITY = 1_000.0
RISK_PCT = 0.02
LEVERAGE = 10

Simulator._load_funding = lambda self, engine: {}


# ── Data loading (reuses parquet cache from compare_breakeven) ──────────

def fetch_5m_candles(symbol: str, since: str, until: str) -> pd.DataFrame:
    import ccxt
    exchange = ccxt.bybit({"enableRateLimit": True})
    exchange.load_markets()
    from src.live.order_manager import _to_ccxt_symbol
    bybit_symbol = _to_ccxt_symbol(symbol)
    if bybit_symbol not in exchange.markets:
        bybit_symbol = symbol.replace("USDT", "/USDT:USDT")

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
    candle_cache = {}
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
        candle_cache[sym] = df
    return candle_cache


# ── Run single backtest, return trades ──────────────────────────────────

def run_backtest(symbol, tf, trading_bars, context_df, from_date, to_date):
    strat = SniperStrategy()
    adapter = StrategyAdapter(strat, None, symbol, from_date, to_date, context_df=context_df)
    sim = Simulator(
        strategy=adapter, symbol=symbol,
        leverage=LEVERAGE, risk_pct=RISK_PCT,
        equity=EQUITY, fixed_risk=True,
    )
    return sim.run(None, from_date, to_date, bars=trading_bars)


def summarize(trades: list[ClosedTrade], label: str = "") -> dict:
    if not trades:
        return {"label": label, "trades": 0, "pnl": 0, "wr": 0,
                "avg_win": 0, "avg_loss": 0, "sharpe": 0, "max_dd": 0}
    usd = compute_pnl_dollar_summary(trades, EQUITY)
    wins = sum(1 for t in trades if t.win_loss)
    return {
        "label": label,
        "trades": len(trades),
        "pnl": usd["total_pnl_usd"],
        "wr": wins / len(trades),
        "avg_win": usd["avg_win_usd"],
        "avg_loss": usd["avg_loss_usd"],
        "sharpe": compute_sharpe([t.pnl_pct for t in trades],
                                 len(trades) / 3 * 365 / max(len(trades), 1)),
        "max_dd": usd["max_dd_vs_peak_pct"],
    }


# ═══════════════════════════════════════════════════════════════════════
# PART 1: Symbol deep-dive
# ═══════════════════════════════════════════════════════════════════════

def deep_dive(all_trades: dict[str, list[ClosedTrade]]):
    print(f"\n{'='*90}")
    print(f"  PART 1: SYMBOL DEEP DIVE -- Why does each symbol win or lose?")
    print(f"{'='*90}")

    for key, trades in sorted(all_trades.items()):
        sym, tf = key.split("|")
        if not trades:
            continue

        usd = compute_pnl_dollar_summary(trades, EQUITY)
        wins = sum(1 for t in trades if t.win_loss)
        total = len(trades)

        print(f"\n  {'-'*80}")
        print(f"  {sym} ({tf})   |   {total} trades   |   "
              f"PnL: {usd['total_pnl_usd']:+,.2f}   |   "
              f"WR: {wins/total:.1%}   |   MaxDD: {usd['max_dd_vs_peak_pct']:.1%}")
        print(f"  {'-'*80}")

        # By year
        by_year: dict[int, list[ClosedTrade]] = defaultdict(list)
        for t in trades:
            by_year[t.entry_time.year].append(t)

        print(f"  {'Year':<8} {'Trades':>6} {'Wins':>5} {'WR':>6} "
              f"{'PnL $':>10} {'AvgWin':>8} {'AvgLoss':>8}")
        for yr in sorted(by_year):
            yt = by_year[yr]
            w = sum(1 for t in yt if t.win_loss)
            pnl = sum(t.pnl_usd for t in yt)
            aw = np.mean([t.pnl_usd for t in yt if t.win_loss]) if w else 0
            al = np.mean([t.pnl_usd for t in yt if not t.win_loss]) if (len(yt)-w) else 0
            print(f"  {yr:<8} {len(yt):>6} {w:>5} {w/len(yt):>6.1%} "
                  f"{pnl:>+10,.2f} {aw:>+8.2f} {al:>+8.2f}")

        # By direction
        longs = [t for t in trades if t.direction == "long"]
        shorts = [t for t in trades if t.direction == "short"]
        print(f"\n  {'Dir':<8} {'Trades':>6} {'WR':>6} {'PnL $':>10}")
        for label, subset in [("Long", longs), ("Short", shorts)]:
            if subset:
                w = sum(1 for t in subset if t.win_loss)
                pnl = sum(t.pnl_usd for t in subset)
                print(f"  {label:<8} {len(subset):>6} {w/len(subset):>6.1%} {pnl:>+10,.2f}")

        # By session
        print(f"\n  {'Session':<12} {'Trades':>6} {'WR':>6} {'PnL $':>10}")
        by_session: dict[str, list[ClosedTrade]] = defaultdict(list)
        for t in trades:
            by_session[t.regime_time_of_day].append(t)
        for sess in ["asia", "london", "new_york", "off_hours"]:
            st = by_session.get(sess, [])
            if st:
                w = sum(1 for t in st if t.win_loss)
                pnl = sum(t.pnl_usd for t in st)
                print(f"  {sess:<12} {len(st):>6} {w/len(st):>6.1%} {pnl:>+10,.2f}")

        # By volatility regime
        print(f"\n  {'Volatility':<12} {'Trades':>6} {'WR':>6} {'PnL $':>10}")
        by_vol: dict[str, list[ClosedTrade]] = defaultdict(list)
        for t in trades:
            by_vol[t.regime_volatility].append(t)
        for vol in ["low", "medium", "high"]:
            vt = by_vol.get(vol, [])
            if vt:
                w = sum(1 for t in vt if t.win_loss)
                pnl = sum(t.pnl_usd for t in vt)
                print(f"  {vol:<12} {len(vt):>6} {w/len(vt):>6.1%} {pnl:>+10,.2f}")

        # Biggest winners & losers
        sorted_trades = sorted(trades, key=lambda t: t.pnl_usd)
        worst3 = sorted_trades[:3]
        best3 = sorted_trades[-3:]
        print(f"\n  Top 3 Wins:")
        for t in reversed(best3):
            print(f"    {t.entry_time:%Y-%m-%d %H:%M} {t.direction:>5} "
                  f"PnL {t.pnl_usd:>+8.2f}  exit={t.exit_reason}")
        print(f"  Worst 3 Losses:")
        for t in worst3:
            print(f"    {t.entry_time:%Y-%m-%d %H:%M} {t.direction:>5} "
                  f"PnL {t.pnl_usd:>+8.2f}  exit={t.exit_reason}")


# ═══════════════════════════════════════════════════════════════════════
# PART 2: Partial take-profit simulation
# ═══════════════════════════════════════════════════════════════════════

def run_partial_tp(symbol, tf, trading_bars, context_df, from_date, to_date,
                   tp1_mult_ratio=0.5):
    """
    Simulate partial TP: split each signal into two half-sized trades.
    Leg A: 50% size, TP at 3x ATR (half of 6x)
    Leg B: 50% size, TP at 6x ATR (original)
    """
    strat = SniperStrategy()
    adapter = StrategyAdapter(strat, None, symbol, from_date, to_date, context_df=context_df)

    sim_a = Simulator(strategy=adapter, symbol=symbol,
                      leverage=LEVERAGE, risk_pct=RISK_PCT * 0.5,
                      equity=EQUITY, fixed_risk=True)

    strat_b = SniperStrategy(tp_atr_mult=3.0)
    adapter_b = StrategyAdapter(strat_b, None, symbol, from_date, to_date, context_df=context_df)
    sim_b = Simulator(strategy=adapter_b, symbol=symbol,
                      leverage=LEVERAGE, risk_pct=RISK_PCT * 0.5,
                      equity=EQUITY, fixed_risk=True)

    trades_full_tp = sim_a.run(None, from_date, to_date, bars=trading_bars)
    trades_half_tp = sim_b.run(None, from_date, to_date, bars=trading_bars)

    combined_pnl = sum(t.pnl_usd for t in trades_full_tp) + sum(t.pnl_usd for t in trades_half_tp)
    tp_hits_a = sum(1 for t in trades_full_tp if t.exit_reason == "tp")
    tp_hits_b = sum(1 for t in trades_half_tp if t.exit_reason == "tp")

    return {
        "trades_full": len(trades_full_tp),
        "trades_half": len(trades_half_tp),
        "pnl_full_leg": sum(t.pnl_usd for t in trades_full_tp),
        "pnl_half_leg": sum(t.pnl_usd for t in trades_half_tp),
        "combined_pnl": combined_pnl,
        "tp_full": tp_hits_a,
        "tp_half": tp_hits_b,
    }


def partial_tp_analysis(candle_cache, symbols, timeframes, from_date, to_date):
    print(f"\n{'='*90}")
    print(f"  PART 2: PARTIAL TAKE-PROFIT")
    print(f"  Strategy: close 50% at 3x ATR, let 50% ride to 6x ATR")
    print(f"{'='*90}")

    total_normal = 0.0
    total_partial = 0.0

    hdr = (f"  {'Symbol':<16} {'TF':>4}  {'Normal PnL':>12}  "
           f"{'Partial PnL':>12}  {'Delta':>10}  "
           f"{'TP@3x':>5}  {'TP@6x':>5}")
    print(hdr)
    print(f"  {'-'*80}")

    for sym in symbols:
        df_5m = candle_cache[sym]
        if df_5m.empty:
            continue
        for tf in timeframes:
            trading_bars = build_bars_for_tf(df_5m, tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            context_df = build_bars_for_tf(df_5m, ctx_tf) if ctx_tf != tf else trading_bars

            normal_trades = run_backtest(sym, tf, trading_bars, context_df, from_date, to_date)
            normal_pnl = sum(t.pnl_usd for t in normal_trades) if normal_trades else 0

            partial = run_partial_tp(sym, tf, trading_bars, context_df, from_date, to_date)

            delta = partial["combined_pnl"] - normal_pnl
            total_normal += normal_pnl
            total_partial += partial["combined_pnl"]

            print(f"  {sym:<16} {tf:>4}  {normal_pnl:>+12,.2f}  "
                  f"{partial['combined_pnl']:>+12,.2f}  {delta:>+10,.2f}  "
                  f"{partial['tp_half']:>5}  {partial['tp_full']:>5}")

    print(f"\n  {'TOTAL':<16} {'':>4}  {total_normal:>+12,.2f}  "
          f"{total_partial:>+12,.2f}  {total_partial - total_normal:>+10,.2f}")


# ═══════════════════════════════════════════════════════════════════════
# PART 3: Max concurrent positions
# ═══════════════════════════════════════════════════════════════════════

def max_positions_analysis(candle_cache, symbols, timeframes, from_date, to_date,
                           max_pos_list=(2, 3, 4)):
    """
    Collect all trade entry/exit times across symbols, replay chronologically,
    and skip entries when open count >= max_positions.
    """
    print(f"\n{'='*90}")
    print(f"  PART 3: MAX CONCURRENT POSITIONS CAP")
    print(f"{'='*90}")

    all_trades: list[ClosedTrade] = []
    for sym in symbols:
        df_5m = candle_cache[sym]
        if df_5m.empty:
            continue
        for tf in timeframes:
            trading_bars = build_bars_for_tf(df_5m, tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            context_df = build_bars_for_tf(df_5m, ctx_tf) if ctx_tf != tf else trading_bars
            trades = run_backtest(sym, tf, trading_bars, context_df, from_date, to_date)
            all_trades.extend(trades)

    if not all_trades:
        print("  No trades to analyze.")
        return

    all_trades.sort(key=lambda t: t.entry_time)
    uncapped_pnl = sum(t.pnl_usd for t in all_trades)
    uncapped_count = len(all_trades)

    print(f"\n  Uncapped: {uncapped_count} trades, PnL {uncapped_pnl:>+10,.2f}\n")

    hdr = (f"  {'MaxPos':>6}  {'Trades':>6}  {'Skipped':>7}  "
           f"{'PnL $':>12}  {'Delta':>10}  "
           f"{'WR':>6}  {'MaxDD':>7}")
    print(hdr)
    print(f"  {'-'*70}")

    for max_pos in max_pos_list:
        open_positions: list[ClosedTrade] = []
        kept: list[ClosedTrade] = []
        skipped = 0

        for trade in all_trades:
            open_positions = [p for p in open_positions if p.exit_time > trade.entry_time]
            if len(open_positions) >= max_pos:
                skipped += 1
                continue
            kept.append(trade)
            open_positions.append(trade)

        pnl = sum(t.pnl_usd for t in kept)
        wins = sum(1 for t in kept if t.win_loss)
        wr = wins / len(kept) if kept else 0

        equity_curve = [EQUITY]
        for t in kept:
            equity_curve.append(equity_curve[-1] + t.pnl_usd)
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        delta = pnl - uncapped_pnl
        print(f"  {max_pos:>6}  {len(kept):>6}  {skipped:>7}  "
              f"{pnl:>+12,.2f}  {delta:>+10,.2f}  "
              f"{wr:>6.1%}  {max_dd:>6.1%}")

    # Show peak concurrent positions over time (uncapped)
    events = []
    for t in all_trades:
        events.append((t.entry_time, +1, t.symbol))
        events.append((t.exit_time, -1, t.symbol))
    events.sort(key=lambda e: e[0])

    concurrent = 0
    max_concurrent = 0
    max_concurrent_time = None
    concurrent_histogram: dict[int, int] = defaultdict(int)
    last_ts = events[0][0] if events else None

    for ts, delta, sym in events:
        concurrent += delta
        if concurrent > max_concurrent:
            max_concurrent = concurrent
            max_concurrent_time = ts
        if concurrent >= 0:
            concurrent_histogram[concurrent] += 1

    print(f"\n  Peak concurrent positions: {max_concurrent} "
          f"(at {max_concurrent_time})")
    print(f"  Concurrency distribution:")
    for n in sorted(concurrent_histogram):
        if n > 0:
            print(f"    {n} positions: {concurrent_histogram[n]:>5} events")


# ═══════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Deep strategy analysis")
    p.add_argument("--from", dest="from_date", default="2023-04-01")
    p.add_argument("--to", dest="to_date",
                   default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--symbols", nargs="+", default=ALL_SYMBOLS)
    p.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    args = p.parse_args()

    print(f"\n{'='*90}")
    print(f"  DEEP STRATEGY ANALYSIS")
    print(f"  {args.from_date} -> {args.to_date}")
    print(f"  Symbols:    {', '.join(args.symbols)}")
    print(f"  Timeframes: {', '.join(args.timeframes)}")
    print(f"{'='*90}\n")

    t_start = time.time()
    candle_cache = load_candles(args.symbols, args.from_date, args.to_date)

    # Collect all trades for deep dive
    all_trades: dict[str, list[ClosedTrade]] = {}
    for sym in args.symbols:
        df_5m = candle_cache[sym]
        if df_5m.empty:
            continue
        for tf in args.timeframes:
            trading_bars = build_bars_for_tf(df_5m, tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            context_df = build_bars_for_tf(df_5m, ctx_tf) if ctx_tf != tf else trading_bars
            trades = run_backtest(sym, tf, trading_bars, context_df, args.from_date, args.to_date)
            all_trades[f"{sym}|{tf}"] = trades

    # Part 1
    deep_dive(all_trades)

    # Part 2
    partial_tp_analysis(candle_cache, args.symbols, args.timeframes,
                        args.from_date, args.to_date)

    # Part 3
    max_positions_analysis(candle_cache, args.symbols, args.timeframes,
                           args.from_date, args.to_date,
                           max_pos_list=[2, 3, 4, 6])

    print(f"\n  Total elapsed: {time.time() - t_start:.0f}s\n")


if __name__ == "__main__":
    main()
