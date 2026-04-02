"""
Scan altcoins with the Sniper strategy — download from Bybit, backtest in memory.

Downloads 5m candles via ccxt, resamples to 15m/4h, computes indicators,
and runs the Sniper strategy backtest without requiring a database.

Usage
-----
    python scripts/scan_altcoins.py
    python scripts/scan_altcoins.py --from 2023-01-01 --to 2026-04-01
    python scripts/scan_altcoins.py --symbols AVAXUSDT LINKUSDT
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.backtest.simulator import Simulator
from src.backtest.run_backtest import compute_pnl_dollar_summary, compute_sharpe
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies import StrategyAdapter
from src.strategies.strategy_sniper import SniperStrategy

log = structlog.get_logger()

ALTCOINS = [
    "DOGEUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT",
    "ADAUSDT", "DOTUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "SUIUSDT", "1000PEPEUSDT", "WIFUSDT",
    "LTCUSDT", "MATICUSDT", "FILUSDT", "AAVEUSDT",
    "OPUSDT", "INJUSDT", "FETUSDT", "RUNEUSDT",
]

TIMEFRAMES = ["15m", "4h"]

EQUITY = 1_000.0
RISK_PCT = 0.02
LEVERAGE = 10


def fetch_5m_candles(symbol: str, since: str, until: str) -> pd.DataFrame:
    """Download 5m candles from Bybit using ccxt."""
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

    all_candles = []
    current = since_ts
    limit = 1000

    while current < until_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(bybit_symbol, "5m", since=current, limit=limit)
        except Exception as e:
            print(f"  [WARN] {symbol} fetch error at {current}: {e}", flush=True)
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

        if len(all_candles) % 50_000 == 0:
            print(f"    {len(all_candles):>8,} candles ...", flush=True)

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


def _no_funding(self, engine):
    """Patched _load_funding that returns empty dict (no DB needed)."""
    return {}


Simulator._load_funding = _no_funding


def run_backtest(symbol: str, tf: str, trading_bars, context_df, args) -> dict | None:
    """Run a single symbol/tf combo through the Sniper strategy."""
    base = SniperStrategy()
    adapter = StrategyAdapter(
        base, None, symbol, args.from_date, args.to_date,
        context_df=context_df,
    )
    sim = Simulator(
        strategy=adapter,
        symbol=symbol,
        leverage=LEVERAGE,
        risk_pct=RISK_PCT,
        equity=EQUITY,
        fixed_risk=True,
    )

    trades = sim.run(None, args.from_date, args.to_date, bars=trading_bars)

    if not trades:
        return None

    usd = compute_pnl_dollar_summary(trades, EQUITY)
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = sum(1 for t in trades if t.win_loss)
    total = len(trades)
    wr = wins / total if total else 0.0

    d0 = datetime.fromisoformat(args.from_date)
    d1 = datetime.fromisoformat(args.to_date)
    days = max((d1 - d0).days, 1)
    trades_per_year = total / days * 365
    sharpe = compute_sharpe(pnl_pcts, trades_per_year)

    return {
        "symbol": symbol,
        "tf": tf,
        "trades": total,
        "wr": wr,
        "payoff": usd["payoff_ratio"],
        "pnl": usd["total_pnl_usd"],
        "ret": usd["total_pnl_usd"] / EQUITY if EQUITY else 0,
        "sharpe": sharpe,
        "max_dd_pct": usd["max_dd_vs_peak_pct"],
        "pf": usd["profit_factor"],
    }


def parse_args():
    p = argparse.ArgumentParser(description="Scan altcoins with Sniper strategy")
    p.add_argument("--from", dest="from_date", default="2023-01-01")
    p.add_argument("--to", dest="to_date",
                    default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--symbols", nargs="+", default=ALTCOINS)
    p.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    return p.parse_args()


def main():
    args = parse_args()
    results: list[dict] = []
    skipped: list[str] = []

    print(f"\n{'='*70}", flush=True)
    print(f"  SNIPER ALTCOIN SCAN", flush=True)
    print(f"  {args.from_date} -> {args.to_date}", flush=True)
    print(f"  Timeframes: {', '.join(args.timeframes)}", flush=True)
    print(f"  Symbols: {len(args.symbols)}", flush=True)
    print(f"  Equity: ${EQUITY:,.0f}  |  Risk: {RISK_PCT:.0%}  |  Fixed sizing", flush=True)
    print(f"{'='*70}\n", flush=True)

    for idx, symbol in enumerate(args.symbols, 1):
        t0 = time.time()
        print(f"[{idx}/{len(args.symbols)}] {symbol} -- downloading 5m candles ...",
              end=" ", flush=True)

        candles_5m = fetch_5m_candles(symbol, args.from_date, args.to_date)
        if candles_5m.empty or len(candles_5m) < 500:
            print(f"SKIP (only {len(candles_5m)} candles)", flush=True)
            skipped.append(symbol)
            continue
        print(f"{len(candles_5m):,} candles ({time.time()-t0:.0f}s)", flush=True)

        built: dict[str, pd.DataFrame] = {}
        needed_tfs = set(args.timeframes)
        for tf in args.timeframes:
            ctx = CONTEXT_TF.get(tf, tf)
            needed_tfs.add(ctx)

        for tf in sorted(needed_tfs):
            if tf == "5m":
                continue
            built[tf] = build_bars_for_tf(candles_5m, tf)

        for tf in args.timeframes:
            trading_bars = built.get(tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            context_df = built.get(ctx_tf, trading_bars)

            if trading_bars is None or trading_bars.empty:
                continue

            t1 = time.time()
            res = run_backtest(symbol, tf, trading_bars, context_df, args)
            elapsed = time.time() - t1

            if res:
                results.append(res)
                tag = "+++" if res["pnl"] > 0 else "---"
                print(f"  {tf:>3s}: {res['trades']:>4} trades  "
                      f"WR {res['wr']:5.1%}  "
                      f"PnL {res['pnl']:>+10,.2f}  "
                      f"PF {res['pf']:.2f}  "
                      f"{tag}  ({elapsed:.1f}s)", flush=True)
            else:
                print(f"  {tf:>3s}:    0 trades  ({elapsed:.1f}s)", flush=True)

    # ---- Summary table ----
    print(f"\n{'='*90}")
    print(f"  SNIPER ALTCOIN SCAN RESULTS")
    print(f"  {args.from_date} -> {args.to_date}")
    print(f"{'='*90}")
    hdr = (f"{'Symbol':<12} {'TF':>4}  {'Trades':>6}  {'WR':>6}  "
           f"{'Payoff':>7}  {'PnL $':>10}  {'Return':>8}  "
           f"{'Sharpe':>6}  {'MaxDD':>7}  {'PF':>5}")
    print(hdr)
    print("-" * len(hdr))

    sorted_results = sorted(results, key=lambda r: r["pnl"], reverse=True)
    for r in sorted_results:
        pf_str = f"{r['pf']:.2f}" if not math.isinf(r["pf"]) else "inf"
        print(
            f"{r['symbol']:<12} {r['tf']:>4}  {r['trades']:>6}  "
            f"{r['wr']:>5.1%}  {r['payoff']:>6.2f}x  "
            f"{r['pnl']:>+10,.2f}  {r['ret']:>+7.1%}  "
            f"{r['sharpe']:>6.2f}  {r['max_dd_pct']:>6.1%}  "
            f"{pf_str:>5}"
        )

    # ---- Profitable combos ----
    profitable = [r for r in sorted_results if r["pnl"] > 0 and r["pf"] > 1.2]
    if profitable:
        print(f"\n{'='*90}")
        print(f"  PROFITABLE COMBOS (PnL > 0 and PF > 1.2)")
        print(f"{'='*90}")
        for r in profitable:
            pf_str = f"{r['pf']:.2f}" if not math.isinf(r["pf"]) else "inf"
            print(f"  {r['symbol']:<12} {r['tf']:>4}  "
                  f"{r['trades']:>4} trades  WR {r['wr']:5.1%}  "
                  f"PnL {r['pnl']:>+10,.2f}  PF {pf_str}")
    else:
        print("\n  No profitable combos found with PF > 1.2")

    if skipped:
        print(f"\n  Skipped (not on Bybit or insufficient data): {', '.join(skipped)}")

    total_pnl = sum(r["pnl"] for r in results) if results else 0
    total_trades = sum(r["trades"] for r in results) if results else 0
    print(f"\n  Total: {len(results)} combos, {total_trades} trades, "
          f"combined PnL: ${total_pnl:+,.2f}")


if __name__ == "__main__":
    main()
