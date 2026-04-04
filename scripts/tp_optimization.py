"""
TP / Trailing-Stop Optimization for Sniper strategy.

Tests multiple scenarios to find the best exit strategy:
  1. Baseline:           SL=1.2 ATR, TP=6.0 ATR (current)
  2. Tighter TP:         TP=5.0, 4.5, 4.0, 3.5 ATR
  3. Breakeven @25%:     SL moves to entry at 25% unrealized gain
  4. Trailing stop:      Trail activates at X% gain, follows HWM with offset
  5. Combined:           Tighter TP + trailing

Usage
-----
    python scripts/tp_optimization.py
    python scripts/tp_optimization.py --symbols SOLUSDT DOGEUSDT
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

DEFAULT_SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT", "DOTUSDT",
    "1000PEPEUSDT", "DOGEUSDT", "OPUSDT", "NEARUSDT", "SUIUSDT",
]
TIMEFRAMES = ["15m", "4h"]
EQUITY = 10_000.0
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


def run_variant(symbol, tf, trading_bars, context_df, from_date, to_date,
                tp_mult=6.0, breakeven_pct=None,
                trail_after_pct=None, trail_offset_pct=0.0):
    strat = SniperStrategy(tp_atr_mult=tp_mult)
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
    if trail_after_pct is not None:
        sim.trail_after_pct = trail_after_pct
        sim.trail_offset_pct = trail_offset_pct

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

    tp_count = sum(1 for t in trades if t.exit_reason == "tp")
    sl_count = sum(1 for t in trades if t.exit_reason == "sl")
    be_count = sum(1 for t in trades if t.exit_reason == "be")
    trail_count = sum(1 for t in trades if t.exit_reason == "trail")

    near_miss = sum(1 for t in trades
                    if t.exit_reason == "sl" and t.mfe_pct > 0.6 * (tp_mult * 1.2 / 100))

    return {
        "trades": total, "wr": wr,
        "pnl": usd["total_pnl_usd"],
        "sharpe": sharpe,
        "max_dd_pct": usd["max_dd_vs_peak_pct"],
        "pf": usd["profit_factor"],
        "avg_win": usd["avg_win_usd"],
        "avg_loss": usd["avg_loss_usd"],
        "tp": tp_count, "sl": sl_count,
        "be": be_count, "trail": trail_count,
        "near_miss": near_miss,
    }


SCENARIOS = [
    {"label": "Baseline (TP=6.0)",  "tp": 6.0},
    {"label": "TP=5.0",            "tp": 5.0},
    {"label": "TP=4.5",            "tp": 4.5},
    {"label": "TP=4.0",            "tp": 4.0},
    {"label": "TP=3.5",            "tp": 3.5},
    {"label": "TP=3.0",            "tp": 3.0},
    {"label": "BE@25% + TP=6.0",   "tp": 6.0, "be": 0.25},
    {"label": "BE@25% + TP=5.0",   "tp": 5.0, "be": 0.25},
    {"label": "BE@25% + TP=4.0",   "tp": 4.0, "be": 0.25},
    {"label": "Trail@15%/5% TP6",  "tp": 6.0, "trail": 0.15, "trail_off": 0.05},
    {"label": "Trail@10%/3% TP6",  "tp": 6.0, "trail": 0.10, "trail_off": 0.03},
    {"label": "Trail@20%/5% TP6",  "tp": 6.0, "trail": 0.20, "trail_off": 0.05},
    {"label": "Trail@15%/3% TP5",  "tp": 5.0, "trail": 0.15, "trail_off": 0.03},
    {"label": "Trail@10%/3% TP4",  "tp": 4.0, "trail": 0.10, "trail_off": 0.03},
]


def main():
    p = argparse.ArgumentParser(description="TP / Trailing-Stop Optimization")
    p.add_argument("--from", dest="from_date", default="2023-01-01")
    p.add_argument("--to", dest="to_date",
                   default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    args = p.parse_args()

    print(f"\n{'='*100}")
    print(f"  TP / TRAILING-STOP OPTIMIZATION")
    print(f"  {args.from_date} -> {args.to_date}")
    print(f"  Symbols:    {', '.join(args.symbols)}")
    print(f"  Timeframes: {', '.join(args.timeframes)}")
    print(f"  Equity: ${EQUITY:,.0f}  |  Risk: {RISK_PCT:.0%}")
    print(f"  Scenarios:  {len(SCENARIOS)}")
    print(f"{'='*100}\n")

    t_start = time.time()

    cache_dir = project_root / "data_cache"
    cache_dir.mkdir(exist_ok=True)

    candle_cache: dict[str, pd.DataFrame] = {}
    for idx, sym in enumerate(args.symbols):
        cache_path = cache_dir / f"{sym}_{args.from_date}_{args.to_date}_5m.parquet"
        if cache_path.exists():
            print(f"[{idx+1}/{len(args.symbols)}] {sym} -- cache ...", end=" ", flush=True)
            df = pd.read_parquet(cache_path)
        else:
            print(f"[{idx+1}/{len(args.symbols)}] {sym} -- downloading ...", end=" ", flush=True)
            df = fetch_5m_candles(sym, args.from_date, args.to_date)
            if not df.empty:
                df.to_parquet(cache_path)
        print(f"{len(df):,} candles", flush=True)
        candle_cache[sym] = df

    bar_cache: dict[str, dict] = {}
    for sym in args.symbols:
        df_5m = candle_cache[sym]
        if df_5m.empty:
            continue
        bar_cache[sym] = {}
        for tf in args.timeframes:
            trading_bars = build_bars_for_tf(df_5m, tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            context_df = build_bars_for_tf(df_5m, ctx_tf) if ctx_tf != tf else trading_bars
            bar_cache[sym][tf] = (trading_bars, context_df)

    # Run all scenarios
    agg: dict[str, dict] = {}
    for sc in SCENARIOS:
        label = sc["label"]
        agg[label] = {"pnl": 0, "trades": 0, "tp": 0, "sl": 0, "be": 0,
                       "trail": 0, "wins": 0, "near_miss": 0, "results": []}

    total_runs = len(args.symbols) * len(args.timeframes) * len(SCENARIOS)
    run_count = 0

    for sym in args.symbols:
        if sym not in bar_cache:
            continue
        for tf in args.timeframes:
            if tf not in bar_cache[sym]:
                continue
            trading_bars, context_df = bar_cache[sym][tf]

            for sc in SCENARIOS:
                run_count += 1
                label = sc["label"]
                print(f"\r  [{run_count}/{total_runs}] {sym} {tf} {label}",
                      end="          ", flush=True)

                r = run_variant(
                    sym, tf, trading_bars, context_df,
                    args.from_date, args.to_date,
                    tp_mult=sc.get("tp", 6.0),
                    breakeven_pct=sc.get("be"),
                    trail_after_pct=sc.get("trail"),
                    trail_offset_pct=sc.get("trail_off", 0.0),
                )
                if r:
                    agg[label]["pnl"] += r["pnl"]
                    agg[label]["trades"] += r["trades"]
                    agg[label]["tp"] += r["tp"]
                    agg[label]["sl"] += r["sl"]
                    agg[label]["be"] += r["be"]
                    agg[label]["trail"] += r["trail"]
                    agg[label]["wins"] += int(r["wr"] * r["trades"])
                    agg[label]["near_miss"] += r["near_miss"]
                    r["symbol"] = sym
                    r["tf"] = tf
                    agg[label]["results"].append(r)

    print("\r" + " " * 80 + "\r", end="", flush=True)

    # ── Results Table ────────────────────────────────────────
    print(f"\n{'='*110}")
    print(f"  AGGREGATE RESULTS (all symbols, all timeframes)")
    print(f"{'='*110}")
    hdr = (f"{'Scenario':<24} {'Trades':>6} {'WR':>6} {'PnL $':>11} "
           f"{'TP':>5} {'SL':>5} {'BE':>4} {'Trail':>5} "
           f"{'NearMiss':>8} {'vs Base':>10}")
    print(hdr)
    print("-" * len(hdr))

    baseline_pnl = agg[SCENARIOS[0]["label"]]["pnl"]

    for sc in SCENARIOS:
        label = sc["label"]
        a = agg[label]
        wr = a["wins"] / a["trades"] if a["trades"] else 0
        delta = a["pnl"] - baseline_pnl
        delta_str = f"{delta:>+10,.2f}" if label != SCENARIOS[0]["label"] else f"{'--':>10}"
        print(f"{label:<24} {a['trades']:>6} {wr:>6.1%} {a['pnl']:>+11,.2f} "
              f"{a['tp']:>5} {a['sl']:>5} {a['be']:>4} {a['trail']:>5} "
              f"{a['near_miss']:>8} {delta_str}")

    # ── Per-symbol breakdown for top scenarios ───────────────
    best_scenarios = sorted(agg.items(), key=lambda x: x[1]["pnl"], reverse=True)[:3]
    print(f"\n{'='*110}")
    print(f"  TOP 3 SCENARIOS -- per-symbol breakdown")
    print(f"{'='*110}")

    for label, _ in best_scenarios:
        print(f"\n  >> {label}")
        results = agg[label]["results"]
        hdr2 = f"    {'Symbol':<16} {'TF':>4} {'Trades':>6} {'WR':>6} {'PnL $':>10} {'TP':>4} {'SL':>4} {'Trail':>5}"
        print(hdr2)
        print("    " + "-" * (len(hdr2) - 4))
        for r in sorted(results, key=lambda x: x["pnl"], reverse=True):
            print(f"    {r['symbol']:<16} {r['tf']:>4} {r['trades']:>6} "
                  f"{r['wr']:>6.1%} {r['pnl']:>+10,.2f} "
                  f"{r['tp']:>4} {r['sl']:>4} {r.get('trail', 0):>5}")

    # ── Near-miss analysis ──────────────────────────────────
    baseline_results = agg[SCENARIOS[0]["label"]]["results"]
    total_near = agg[SCENARIOS[0]["label"]]["near_miss"]
    total_sl = agg[SCENARIOS[0]["label"]]["sl"]
    print(f"\n{'='*110}")
    print(f"  NEAR-MISS ANALYSIS (baseline TP=6.0)")
    print(f"{'='*110}")
    print(f"  Trades that reached >60% of TP distance but hit SL: {total_near}")
    print(f"  Total SL exits: {total_sl}")
    if total_sl > 0:
        print(f"  Near-miss ratio: {total_near/total_sl:.1%} of all losses were near-misses")

    elapsed = time.time() - t_start
    print(f"\n  Total elapsed: {elapsed:.0f}s\n")


if __name__ == "__main__":
    main()
