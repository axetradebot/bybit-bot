"""
Compare Sniper strategy with and without breakeven stop.

When breakeven_pct is set (e.g. 0.10), the simulator moves the stop-loss
to entry price once unrealized gain reaches that threshold.

Usage
-----
    python scripts/compare_breakeven.py
    python scripts/compare_breakeven.py --symbols SOLUSDT 1000PEPEUSDT
    python scripts/compare_breakeven.py --be-pct 0.08
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

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

DEFAULT_SYMBOLS = ["SOLUSDT", "1000PEPEUSDT"]
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


def run_backtest(symbol, tf, trading_bars, context_df, from_date, to_date,
                 breakeven_pct=None):
    strat = SniperStrategy()
    adapter = StrategyAdapter(
        strat, None, symbol, from_date, to_date,
        context_df=context_df,
    )
    sim = Simulator(
        strategy=adapter, symbol=symbol,
        leverage=LEVERAGE, risk_pct=RISK_PCT,
        equity=EQUITY, fixed_risk=True,
        breakeven_pct=breakeven_pct,
    )
    trades = sim.run(None, from_date, to_date, bars=trading_bars)
    if not trades:
        return None

    usd = compute_pnl_dollar_summary(trades, EQUITY)
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = sum(1 for t in trades if t.win_loss)
    total = len(trades)
    wr = wins / total if total else 0.0

    d0 = datetime.fromisoformat(from_date)
    d1 = datetime.fromisoformat(to_date)
    days = max((d1 - d0).days, 1)
    trades_per_year = total / days * 365
    sharpe = compute_sharpe(pnl_pcts, trades_per_year)

    be_count = sum(1 for t in trades if t.exit_reason == "be")
    sl_count = sum(1 for t in trades if t.exit_reason == "sl")
    tp_count = sum(1 for t in trades if t.exit_reason == "tp")

    return {
        "trades": total, "wr": wr,
        "pnl": usd["total_pnl_usd"],
        "sharpe": sharpe,
        "max_dd_pct": usd["max_dd_vs_peak_pct"],
        "pf": usd["profit_factor"],
        "payoff": usd["payoff_ratio"],
        "avg_win": usd["avg_win_usd"],
        "avg_loss": usd["avg_loss_usd"],
        "tp_count": tp_count,
        "sl_count": sl_count,
        "be_count": be_count,
    }


def main():
    p = argparse.ArgumentParser(description="Compare breakeven stop strategy")
    p.add_argument("--from", dest="from_date", default="2023-01-01")
    p.add_argument("--to", dest="to_date",
                   default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    p.add_argument("--be-pct", type=float, default=0.10,
                   help="Breakeven activation threshold (default 0.10 = 10%%)")
    args = p.parse_args()

    print(f"\n{'='*90}")
    print(f"  BREAKEVEN STOP COMPARISON")
    print(f"  {args.from_date} -> {args.to_date}")
    print(f"  Symbols:    {', '.join(args.symbols)}")
    print(f"  Timeframes: {', '.join(args.timeframes)}")
    print(f"  BE trigger: {args.be_pct:.0%} unrealized gain -> SL moves to entry")
    print(f"  Equity: ${EQUITY:,.0f}  |  Risk: {RISK_PCT:.0%}")
    print(f"{'='*90}\n")

    t_start = time.time()

    cache_dir = project_root / "data_cache"
    cache_dir.mkdir(exist_ok=True)

    candle_cache: dict[str, pd.DataFrame] = {}
    for sym in args.symbols:
        cache_path = cache_dir / f"{sym}_{args.from_date}_{args.to_date}_5m.parquet"
        if cache_path.exists():
            print(f"[{args.symbols.index(sym)+1}/{len(args.symbols)}] "
                  f"{sym} -- loading from cache ...", end=" ", flush=True)
            df = pd.read_parquet(cache_path)
        else:
            print(f"[{args.symbols.index(sym)+1}/{len(args.symbols)}] "
                  f"{sym} -- downloading 5m candles ...", end=" ", flush=True)
            df = fetch_5m_candles(sym, args.from_date, args.to_date)
            if not df.empty:
                df.to_parquet(cache_path)
        print(f"{len(df):,} candles", flush=True)
        candle_cache[sym] = df

    results_normal: list[dict] = []
    results_be: list[dict] = []

    for sym in args.symbols:
        df_5m = candle_cache[sym]
        if df_5m.empty:
            continue

        for tf in args.timeframes:
            trading_bars = build_bars_for_tf(df_5m, tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            if ctx_tf != tf:
                context_df = build_bars_for_tf(df_5m, ctx_tf)
            else:
                context_df = trading_bars

            r_normal = run_backtest(
                sym, tf, trading_bars, context_df,
                args.from_date, args.to_date,
                breakeven_pct=None,
            )
            r_be = run_backtest(
                sym, tf, trading_bars, context_df,
                args.from_date, args.to_date,
                breakeven_pct=args.be_pct,
            )

            if r_normal:
                r_normal["symbol"] = sym
                r_normal["tf"] = tf
                results_normal.append(r_normal)
            if r_be:
                r_be["symbol"] = sym
                r_be["tf"] = tf
                results_be.append(r_be)

    # Print per-symbol/tf comparison
    print(f"\n{'='*90}")
    print(f"  PER-SYMBOL RESULTS")
    print(f"{'='*90}")
    hdr = (f"{'Symbol':<16} {'TF':>4}  {'Mode':<12} "
           f"{'Trades':>6}  {'WR':>5}  {'PnL $':>10}  "
           f"{'TP':>4}  {'SL':>4}  {'BE':>4}  "
           f"{'MaxDD':>6}  {'Sharpe':>6}")
    print(hdr)
    print("-" * len(hdr))

    for rn, rb in zip(results_normal, results_be):
        sym = rn["symbol"]
        tf = rn["tf"]
        print(f"{sym:<16} {tf:>4}  {'No BE':<12} "
              f"{rn['trades']:>6}  {rn['wr']:>5.1%}  {rn['pnl']:>+10,.2f}  "
              f"{rn['tp_count']:>4}  {rn['sl_count']:>4}  {rn['be_count']:>4}  "
              f"{rn['max_dd_pct']:>5.1%}  {rn['sharpe']:>6.2f}")
        print(f"{'':<16} {'':>4}  {f'BE@{args.be_pct:.0%}':<12} "
              f"{rb['trades']:>6}  {rb['wr']:>5.1%}  {rb['pnl']:>+10,.2f}  "
              f"{rb['tp_count']:>4}  {rb['sl_count']:>4}  {rb['be_count']:>4}  "
              f"{rb['max_dd_pct']:>5.1%}  {rb['sharpe']:>6.2f}")
        delta = rb["pnl"] - rn["pnl"]
        print(f"{'':<16} {'':>4}  {'Delta':<12} "
              f"{'':>6}  {'':>5}  {delta:>+10,.2f}")
        print()

    # Aggregate totals
    total_n = sum(r["pnl"] for r in results_normal)
    total_b = sum(r["pnl"] for r in results_be)
    trades_n = sum(r["trades"] for r in results_normal)
    trades_b = sum(r["trades"] for r in results_be)
    tp_n = sum(r["tp_count"] for r in results_normal)
    sl_n = sum(r["sl_count"] for r in results_normal)
    tp_b = sum(r["tp_count"] for r in results_be)
    sl_b = sum(r["sl_count"] for r in results_be)
    be_b = sum(r["be_count"] for r in results_be)

    print(f"{'='*90}")
    print(f"  AGGREGATE COMPARISON")
    print(f"{'='*90}")
    print(f"  {'No Breakeven:':<20} {trades_n:>5} trades  "
          f"PnL {total_n:>+10,.2f}  TP: {tp_n}  SL: {sl_n}")
    print(f"  {f'BE @ {args.be_pct:.0%}:':<20} {trades_b:>5} trades  "
          f"PnL {total_b:>+10,.2f}  TP: {tp_b}  SL: {sl_b}  BE: {be_b}")
    print(f"  {'Delta:':<20} {'':>5}        "
          f"PnL {total_b - total_n:>+10,.2f}")

    saved_losses = sl_n - sl_b
    lost_winners = tp_n - tp_b
    print(f"\n  Losses converted to breakeven: {be_b}")
    print(f"  SL exits avoided:              {saved_losses}")
    print(f"  TP exits lost (stopped at BE): {lost_winners}")
    print(f"\n  Elapsed: {time.time() - t_start:.0f}s\n")


if __name__ == "__main__":
    main()
