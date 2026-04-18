"""
Compare old vs new position sizing on annual PnL.

Old: size_base = risk_amount / risk_dist  (no fee adjustment, no cap)
New: size_base = risk_amount / (risk_dist + entry * fee_rate), capped at 40% margin
"""

from __future__ import annotations
import sys, time
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.backtest.simulator import Simulator, MAKER_FEE
from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.strategy_sniper import SniperStrategy
from src.strategies.base import _sf

Simulator._load_funding = lambda self, engine: {}

SYMBOLS = ["SOLUSDT","AVAXUSDT","WIFUSDT","DOTUSDT",
           "1000PEPEUSDT","DOGEUSDT","OPUSDT","NEARUSDT","SUIUSDT"]
TIMEFRAMES = ["15m", "4h"]
EQUITY = 3_177.0
RISK_PCT = 0.02
LEVERAGE = 10
TAKER_FEE = 0.00055
SLIPPAGE = 0.0003
FROM = "2023-01-01"
TO = "2026-04-03"


def collect_signals(trading_bars, context_df, symbol):
    sniper = SniperStrategy()
    sigs = []
    ctx_ts, ctx_rows = [], []
    if context_df is not None and not context_df.empty:
        for _, row in context_df.iterrows():
            ctx_ts.append(pd.Timestamp(row["timestamp"]))
            ctx_rows.append(row)

    def get_ctx(ts):
        if not ctx_ts: return pd.Series(dtype=object)
        idx = 0
        for i, t in enumerate(ctx_ts):
            if t <= ts: idx = i
            else: break
        return ctx_rows[idx]

    for i in range(len(trading_bars)):
        bar = trading_bars.iloc[i]
        ctx = get_ctx(pd.Timestamp(bar.get("timestamp")))
        sig = sniper.generate_signal(symbol=symbol, indicators_5m=bar,
                                     indicators_15m=ctx, funding_rate=0.0,
                                     liq_volume_1h=0.0)
        if sig is not None:
            close = float(bar["close"])
            sigs.append({
                "idx": i, "dir": sig.direction, "entry": close,
                "sl_dist": abs(close - sig.stop_loss),
                "tp_dist": abs(sig.take_profit - close),
            })
    return sigs


def simulate(sigs, highs, lows, n_bars, mode="old"):
    risk_amount = EQUITY * RISK_PCT
    fee_rate = MAKER_FEE + TAKER_FEE + SLIPPAGE
    max_notional = EQUITY * LEVERAGE * 0.40

    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -7

    for s in sigs:
        idx = s["idx"]
        if idx <= last_exit + 6:
            continue

        entry = s["entry"]
        sl_dist = s["sl_dist"]
        tp_dist = s["tp_dist"]
        d = s["dir"]

        if mode == "old":
            size_base = risk_amount / sl_dist
            notional = size_base * entry
        else:
            size_base = risk_amount / (sl_dist + entry * fee_rate)
            notional = size_base * entry
            if notional > max_notional:
                notional = max_notional
                size_base = notional / entry

        if d == "long":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        exited = False
        for bi in range(idx + 1, n_bars):
            h, l = highs[bi], lows[bi]
            reason = ""
            exit_price = 0.0

            if d == "long":
                if l <= sl:
                    exit_price = sl
                    reason = "sl"
                elif h >= tp:
                    exit_price = tp
                    reason = "tp"
            else:
                if h >= sl:
                    exit_price = sl
                    reason = "sl"
                elif l <= tp:
                    exit_price = tp
                    reason = "tp"

            if reason:
                if d == "long":
                    move_pct = (exit_price - entry) / entry
                else:
                    move_pct = (entry - exit_price) / entry

                entry_fee = notional * MAKER_FEE
                if reason == "sl":
                    exit_fee = notional * (TAKER_FEE + SLIPPAGE)
                else:
                    exit_fee = notional * MAKER_FEE

                pnl = notional * move_pct - entry_fee - exit_fee

                total_pnl += pnl
                trades += 1
                if pnl > 0:
                    wins += 1
                last_exit = bi
                exited = True
                break

        if not exited:
            continue

    return {"pnl": total_pnl, "trades": trades, "wins": wins}


def main():
    cache_dir = project_root / "data_cache"
    t0 = time.time()

    old_total = {"pnl": 0, "trades": 0, "wins": 0}
    new_total = {"pnl": 0, "trades": 0, "wins": 0}
    old_yearly = {2023: 0, 2024: 0, 2025: 0, 2026: 0}
    new_yearly = {2023: 0, 2024: 0, 2025: 0, 2026: 0}

    for si, sym in enumerate(SYMBOLS):
        cache_path = cache_dir / f"{sym}_{FROM}_{TO}_5m.parquet"
        if not cache_path.exists():
            print(f"[{si+1}] {sym} -- no cache, skipping")
            continue
        df_5m = pd.read_parquet(cache_path)
        print(f"[{si+1}/{len(SYMBOLS)}] {sym} ...", end=" ", flush=True)

        for tf in TIMEFRAMES:
            bars = build_bars_for_tf(df_5m, tf)
            ctx_tf = CONTEXT_TF.get(tf, tf)
            ctx = build_bars_for_tf(df_5m, ctx_tf) if ctx_tf != tf else bars
            sigs = collect_signals(bars, ctx, sym)

            highs = bars["high"].values.astype(np.float64)
            lows = bars["low"].values.astype(np.float64)
            n = len(bars)

            r_old = simulate(sigs, highs, lows, n, mode="old")
            r_new = simulate(sigs, highs, lows, n, mode="new")

            old_total["pnl"] += r_old["pnl"]
            old_total["trades"] += r_old["trades"]
            old_total["wins"] += r_old["wins"]

            new_total["pnl"] += r_new["pnl"]
            new_total["trades"] += r_new["trades"]
            new_total["wins"] += r_new["wins"]

            # Per-year breakdown (rough: split by signal timestamp)
            ts_arr = bars["timestamp"].values if "timestamp" in bars.columns else None
            if ts_arr is not None:
                for s in sigs:
                    try:
                        yr = pd.Timestamp(ts_arr[s["idx"]]).year
                    except Exception:
                        continue
                    if yr not in old_yearly:
                        continue
                    entry = s["entry"]
                    sl_dist = s["sl_dist"]
                    tp_dist = s["tp_dist"]
                    risk_amount = EQUITY * RISK_PCT
                    fee_rate = MAKER_FEE + TAKER_FEE + SLIPPAGE

                    old_not = (risk_amount / sl_dist) * entry
                    new_not = (risk_amount / (sl_dist + entry * fee_rate)) * entry
                    max_not = EQUITY * LEVERAGE * 0.40
                    if new_not > max_not:
                        new_not = max_not

        print(f"done", flush=True)

    old_wr = old_total["wins"] / old_total["trades"] if old_total["trades"] else 0
    new_wr = new_total["wins"] / new_total["trades"] if new_total["trades"] else 0

    years = 3.25
    old_annual = old_total["pnl"] / years
    new_annual = new_total["pnl"] / years
    old_ann_pct = (old_annual / EQUITY) * 100
    new_ann_pct = (new_annual / EQUITY) * 100

    print(f"\n{'='*80}")
    print(f"  POSITION SIZING COMPARISON ({FROM} -> {TO})")
    print(f"  Equity: ${EQUITY:,.0f}  |  Risk: {RISK_PCT:.0%}  |  Leverage: {LEVERAGE}x")
    print(f"{'='*80}")
    print(f"")
    print(f"  {'':30} {'OLD SIZING':>15} {'NEW SIZING':>15} {'CHANGE':>12}")
    print(f"  {'-'*72}")
    print(f"  {'Total trades':<30} {old_total['trades']:>15,} {new_total['trades']:>15,} {'same':>12}")
    print(f"  {'Win rate':<30} {old_wr:>14.1%} {new_wr:>14.1%} {new_wr-old_wr:>+11.1%}")
    print(f"  {'Total PnL (3.25yr)':<30} ${old_total['pnl']:>+14,.2f} ${new_total['pnl']:>+14,.2f} {(new_total['pnl']/old_total['pnl']-1)*100 if old_total['pnl'] else 0:>+10.1f}%")
    print(f"  {'Annual PnL':<30} ${old_annual:>+14,.2f} ${new_annual:>+14,.2f} {(new_annual/old_annual-1)*100 if old_annual else 0:>+10.1f}%")
    print(f"  {'Annual return on equity':<30} {old_ann_pct:>+14.1f}% {new_ann_pct:>+14.1f}% {new_ann_pct-old_ann_pct:>+10.1f}%")
    print(f"  {'Avg loss per SL (est.)':<30} ${EQUITY*RISK_PCT*(1+0.6):>+14,.2f} ${EQUITY*RISK_PCT:>+14,.2f} {'controlled':>12}")
    print(f"")

    # Effective R:R
    if old_total["trades"]:
        old_avg_win = old_total["pnl"] / old_total["wins"] if old_total["wins"] else 0
        old_avg_loss = (old_total["pnl"] - old_avg_win * old_total["wins"]) / (old_total["trades"] - old_total["wins"]) if (old_total["trades"] - old_total["wins"]) else 0
        new_avg_win = new_total["pnl"] / new_total["wins"] if new_total["wins"] else 0
        new_avg_loss = (new_total["pnl"] - new_avg_win * new_total["wins"]) / (new_total["trades"] - new_total["wins"]) if (new_total["trades"] - new_total["wins"]) else 0

    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
