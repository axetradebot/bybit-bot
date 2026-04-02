"""
Grid-search optimizer for SniperStrategy filter parameters.

Downloads 5m candles from Bybit via ccxt (no database required), resamples
to target timeframes, and sweeps configurable filter thresholds to find the
parameter combination that maximises PnL.

Usage
-----
    python scripts/optimize_sniper.py
    python scripts/optimize_sniper.py --symbols SOLUSDT ARBUSDT --timeframes 15m
    python scripts/optimize_sniper.py --max-combos 300
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import random
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

# ── Default symbols: current live set ───────────────────────────────────
DEFAULT_SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT", "DOTUSDT",
    "1000PEPEUSDT", "ARBUSDT",
]

TIMEFRAMES = ["15m", "4h"]

EQUITY = 1_000.0
RISK_PCT = 0.02
LEVERAGE = 10

# ── Parameter grid ──────────────────────────────────────────────────────
PARAM_GRID = {
    "ema_spread_min": [0.002, 0.003, 0.005, 0.007, 0.01],
    "rsi_long_lo":    [30, 35, 40],
    "rsi_long_hi":    [55, 60, 65],
    "rsi_short_lo":   [35, 40, 45],
    "rsi_short_hi":   [60, 65, 70],
    "atr_rank_floor": [0.10, 0.15, 0.20, 0.25, 0.30],
    "sl_atr_mult":    [1.5, 2.0, 2.5],
    "tp_atr_mult":    [3.0, 4.0, 5.0, 6.0],
    "ema_touch_slack": [0.001, 0.002, 0.003, 0.004],
}

DEFAULT_MAX_COMBOS = 500


# ── Data fetching (reused from scan_altcoins.py) ────────────────────────

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

    all_candles: list[list] = []
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


# ── Monkey-patch Simulator to skip DB funding lookup ────────────────────

def _no_funding(self, engine):
    return {}

Simulator._load_funding = _no_funding


# ── Grid helpers ────────────────────────────────────────────────────────

def build_combos(grid: dict, max_combos: int) -> list[dict]:
    """Cartesian product of grid values, randomly sampled if too large."""
    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    total = 1
    for v in values:
        total *= len(v)

    if total <= max_combos:
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
    else:
        combos = []
        seen: set[tuple] = set()
        while len(combos) < max_combos:
            vals = tuple(random.choice(v) for v in values)
            if vals not in seen:
                seen.add(vals)
                combos.append(dict(zip(keys, vals)))
        # always include the default params
        defaults = {k: PARAM_GRID[k][len(PARAM_GRID[k]) // 2] for k in keys}
        defaults.update(
            ema_spread_min=0.005, rsi_long_lo=35, rsi_long_hi=60,
            rsi_short_lo=40, rsi_short_hi=65, atr_rank_floor=0.20,
            sl_atr_mult=2.0, tp_atr_mult=4.0, ema_touch_slack=0.002,
        )
        if defaults not in combos:
            combos[0] = defaults

    return combos


def params_str(p: dict) -> str:
    return " | ".join(f"{k}={v}" for k, v in sorted(p.items()))


# ── Single backtest run ─────────────────────────────────────────────────

def run_one(
    params: dict,
    symbol: str,
    tf: str,
    trading_bars: pd.DataFrame,
    context_df: pd.DataFrame,
    from_date: str,
    to_date: str,
) -> dict | None:
    """Backtest one parameter combo on one symbol/tf pair."""
    strat = SniperStrategy(**params)
    adapter = StrategyAdapter(
        strat, None, symbol, from_date, to_date,
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

    return {
        "params": params,
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


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Grid-search Sniper strategy filters")
    p.add_argument("--from", dest="from_date", default="2023-01-01")
    p.add_argument("--to", dest="to_date",
                    default=(date.today() - timedelta(days=1)).isoformat())
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    p.add_argument("--max-combos", type=int, default=DEFAULT_MAX_COMBOS,
                    help="Cap on total parameter combos to test")
    return p.parse_args()


# ── Main ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    combos = build_combos(PARAM_GRID, args.max_combos)

    print(f"\n{'='*80}", flush=True)
    print(f"  SNIPER FILTER OPTIMIZER", flush=True)
    print(f"  {args.from_date} -> {args.to_date}", flush=True)
    print(f"  Symbols:    {', '.join(args.symbols)}", flush=True)
    print(f"  Timeframes: {', '.join(args.timeframes)}", flush=True)
    print(f"  Combos:     {len(combos)}", flush=True)
    print(f"  Equity: ${EQUITY:,.0f}  |  Risk: {RISK_PCT:.0%}  |  Fixed sizing", flush=True)
    print(f"{'='*80}\n", flush=True)

    # ── Phase 1: download and build bars per symbol ─────────────────────
    market_data: dict[str, dict[str, pd.DataFrame]] = {}

    for idx, symbol in enumerate(args.symbols, 1):
        t0 = time.time()
        print(f"[{idx}/{len(args.symbols)}] {symbol} -- downloading 5m candles ...",
              end=" ", flush=True)

        candles_5m = fetch_5m_candles(symbol, args.from_date, args.to_date)
        if candles_5m.empty or len(candles_5m) < 500:
            print(f"SKIP (only {len(candles_5m)} candles)", flush=True)
            continue
        print(f"{len(candles_5m):,} candles ({time.time()-t0:.0f}s)", flush=True)

        needed_tfs = set(args.timeframes)
        for tf in args.timeframes:
            ctx = CONTEXT_TF.get(tf, tf)
            needed_tfs.add(ctx)

        built: dict[str, pd.DataFrame] = {}
        for tf in sorted(needed_tfs):
            if tf == "5m":
                continue
            built[tf] = build_bars_for_tf(candles_5m, tf)

        market_data[symbol] = built

    if not market_data:
        print("\n  No data downloaded. Exiting.", flush=True)
        return

    # ── Phase 2: grid-search ────────────────────────────────────────────
    all_results: list[dict] = []
    total_runs = len(combos) * sum(
        len(args.timeframes) for s in args.symbols if s in market_data
    )
    run_count = 0
    t_start = time.time()

    for ci, params in enumerate(combos, 1):
        for symbol, built in market_data.items():
            for tf in args.timeframes:
                trading_bars = built.get(tf)
                ctx_tf = CONTEXT_TF.get(tf, tf)
                context_df = built.get(ctx_tf, trading_bars)

                if trading_bars is None or trading_bars.empty:
                    continue

                run_count += 1
                res = run_one(
                    params, symbol, tf,
                    trading_bars, context_df,
                    args.from_date, args.to_date,
                )
                if res:
                    all_results.append(res)

        if ci % 50 == 0 or ci == len(combos):
            elapsed = time.time() - t_start
            rate = run_count / elapsed if elapsed > 0 else 0
            print(f"  combo {ci}/{len(combos)}  "
                  f"({run_count} runs, {rate:.1f} runs/s, "
                  f"{elapsed:.0f}s elapsed)", flush=True)

    # ── Phase 3: aggregate results per combo ────────────────────────────
    combo_agg: dict[str, dict] = {}
    for r in all_results:
        key = params_str(r["params"])
        if key not in combo_agg:
            combo_agg[key] = {
                "params": r["params"],
                "total_pnl": 0.0,
                "total_trades": 0,
                "wins": 0,
                "details": [],
            }
        combo_agg[key]["total_pnl"] += r["pnl"]
        combo_agg[key]["total_trades"] += r["trades"]
        combo_agg[key]["wins"] += int(r["wr"] * r["trades"])
        combo_agg[key]["details"].append(r)

    ranked = sorted(combo_agg.values(), key=lambda x: x["total_pnl"], reverse=True)

    # ── Console output ──────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"  TOP 20 COMBOS (by aggregate PnL across all symbols/TFs)")
    print(f"{'='*100}")
    hdr = f"{'Rank':>4}  {'Trades':>6}  {'WR':>6}  {'PnL $':>10}  Parameters"
    print(hdr)
    print("-" * len(hdr))

    for i, entry in enumerate(ranked[:20], 1):
        t = entry["total_trades"]
        wr = entry["wins"] / t if t else 0
        print(f"{i:>4}  {t:>6}  {wr:>5.1%}  "
              f"{entry['total_pnl']:>+10,.2f}  "
              f"{params_str(entry['params'])}")

    if len(ranked) > 5:
        print(f"\n{'='*100}")
        print(f"  BOTTOM 5 COMBOS")
        print(f"{'='*100}")
        for entry in ranked[-5:]:
            t = entry["total_trades"]
            wr = entry["wins"] / t if t else 0
            print(f"       {t:>6}  {wr:>5.1%}  "
                  f"{entry['total_pnl']:>+10,.2f}  "
                  f"{params_str(entry['params'])}")

    # ── Per-symbol breakdown for the best combo ─────────────────────────
    if ranked:
        best = ranked[0]
        print(f"\n{'='*100}")
        print(f"  BEST COMBO -- per-symbol breakdown")
        print(f"  {params_str(best['params'])}")
        print(f"{'='*100}")
        bhdr = f"  {'Symbol':<14} {'TF':>4}  {'Trades':>6}  {'WR':>6}  {'PnL $':>10}  {'PF':>5}  {'Sharpe':>6}  {'MaxDD':>7}"
        print(bhdr)
        print("  " + "-" * (len(bhdr) - 2))
        for d in sorted(best["details"], key=lambda x: x["pnl"], reverse=True):
            pf_str = f"{d['pf']:.2f}" if not math.isinf(d["pf"]) else "inf"
            print(f"  {d['symbol']:<14} {d['tf']:>4}  {d['trades']:>6}  "
                  f"{d['wr']:>5.1%}  {d['pnl']:>+10,.2f}  "
                  f"{pf_str:>5}  {d['sharpe']:>6.2f}  {d['max_dd_pct']:>6.1%}")

    # ── Comparison: current defaults vs best ────────────────────────────
    default_key = params_str({
        "atr_rank_floor": 0.20, "ema_spread_min": 0.005,
        "ema_touch_slack": 0.002, "rsi_long_hi": 60, "rsi_long_lo": 35,
        "rsi_short_hi": 65, "rsi_short_lo": 40,
        "sl_atr_mult": 2.0, "tp_atr_mult": 4.0,
    })
    default_entry = combo_agg.get(default_key)
    if default_entry and ranked:
        dt = default_entry["total_trades"]
        dw = default_entry["wins"] / dt if dt else 0
        bt = ranked[0]["total_trades"]
        bw = ranked[0]["wins"] / bt if bt else 0
        print(f"\n{'='*100}")
        print(f"  CURRENT DEFAULTS vs BEST")
        print(f"{'='*100}")
        print(f"  Default:  {dt:>5} trades  WR {dw:>5.1%}  PnL {default_entry['total_pnl']:>+10,.2f}")
        print(f"  Best:     {bt:>5} trades  WR {bw:>5.1%}  PnL {ranked[0]['total_pnl']:>+10,.2f}")
        diff = ranked[0]["total_pnl"] - default_entry["total_pnl"]
        print(f"  Delta:    {diff:>+10,.2f}")

    # ── Save CSV ────────────────────────────────────────────────────────
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"sniper_optimization_{ts_str}.csv"

    fieldnames = [
        "rank", "total_pnl", "total_trades", "wr",
        *sorted(PARAM_GRID.keys()),
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, entry in enumerate(ranked, 1):
            t = entry["total_trades"]
            wr = entry["wins"] / t if t else 0
            row = {
                "rank": i,
                "total_pnl": round(entry["total_pnl"], 2),
                "total_trades": t,
                "wr": round(wr, 4),
                **entry["params"],
            }
            writer.writerow(row)

    print(f"\n  Results saved to {csv_path}")
    print(f"  Total: {len(ranked)} combos, {sum(e['total_trades'] for e in ranked)} total trades")
    print(f"  Elapsed: {time.time() - t_start:.0f}s\n")


if __name__ == "__main__":
    main()
