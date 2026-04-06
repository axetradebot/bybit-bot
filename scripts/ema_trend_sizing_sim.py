"""
EMA trend-aligned position sizing simulation.

Uses EMA 50 / EMA 200 on 15m bars to determine trend direction:
  - EMA 50 > EMA 200 = bullish trend
  - EMA 50 < EMA 200 = bearish trend

Position sizing adjustment:
  - Trade WITH trend  (long in bull, short in bear): boost risk
  - Trade AGAINST trend (long in bear, short in bull): reduce risk
  - No clear trend (EMAs very close / crossed): base risk

Compares multiple boost/reduce combos against flat 1% baseline.

Usage:
    python scripts/ema_trend_sizing_sim.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.strategy_sniper import SniperStrategy

MAKER_FEE = 0.0002
TAKER_FEE = 0.00055
SLIPPAGE_BPS = 0.0003
ROUND_TRIP_COST = MAKER_FEE + TAKER_FEE + SLIPPAGE_BPS
NOTIONAL_CAP_FRAC = 0.25
COOLDOWN_BARS = 6
LEVERAGE = 20
TRAIL_ACTIVATE = 0.10
TRAIL_OFFSET = 0.03
START_EQUITY = 3000.0
BASE_RISK = 0.01
MAX_CONCURRENT = 4

SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT",
    "1000PEPEUSDT", "DOGEUSDT", "OPUSDT", "SUIUSDT",
]
TIMEFRAMES = ["15m", "4h"]
FROM = "2023-01-01"
TO = "2026-04-03"

# Scenarios: (label, with_trend_risk, against_trend_risk, neutral_risk)
SCENARIOS = [
    ("Flat 1.0%",             0.010, 0.010, 0.010),
    ("With 1.25% / Against 0.75%", 0.0125, 0.0075, 0.010),
    ("With 1.5%  / Against 0.5%",  0.015, 0.005, 0.010),
    ("With 1.5%  / Against 0.75%", 0.015, 0.0075, 0.010),
    ("With 1.5%  / Skip against",  0.015, 0.000, 0.010),
    ("With 2.0%  / Against 0.5%",  0.020, 0.005, 0.010),
    ("With 1.25% / Skip against",  0.0125, 0.000, 0.010),
    ("With 1.0%  / Skip against",  0.010, 0.000, 0.010),
]

EMA_NEUTRAL_BAND = 0.001  # EMAs within 0.1% = neutral (no clear trend)


@dataclass
class PrecomputedTrade:
    symbol: str
    tf: str
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    sl_distance: float
    tp_distance: float
    raw_pct: float
    exit_reason: str
    bars_held: int
    ema50: float
    ema200: float


def collect_and_resolve(trading_bars, context_df, symbol, tf):
    sniper = SniperStrategy()
    trades = []

    ctx_ts, ctx_rows = [], []
    if context_df is not None and not context_df.empty:
        for _, row in context_df.iterrows():
            ctx_ts.append(pd.Timestamp(row["timestamp"]))
            ctx_rows.append(row)

    def get_ctx(ts):
        if not ctx_ts:
            return pd.Series(dtype=object)
        idx = 0
        for i, t in enumerate(ctx_ts):
            if t <= ts:
                idx = i
            else:
                break
        return ctx_rows[idx]

    highs = trading_bars["high"].values.astype(np.float64)
    lows = trading_bars["low"].values.astype(np.float64)
    closes = trading_bars["close"].values.astype(np.float64)
    timestamps = trading_bars["timestamp"].values

    ema50_vals = trading_bars["ema_50"].values.astype(np.float64) \
        if "ema_50" in trading_bars.columns else np.full(len(trading_bars), np.nan)
    ema200_vals = trading_bars["ema_200"].values.astype(np.float64) \
        if "ema_200" in trading_bars.columns else np.full(len(trading_bars), np.nan)

    n = len(trading_bars)
    last_exit_idx = -COOLDOWN_BARS - 1

    for i in range(n):
        if i <= last_exit_idx + COOLDOWN_BARS:
            continue

        bar = trading_bars.iloc[i]
        ctx = get_ctx(pd.Timestamp(bar.get("timestamp")))
        sig = sniper.generate_signal(
            symbol=symbol, indicators_5m=bar,
            indicators_15m=ctx, funding_rate=0.0,
            liq_volume_1h=0.0,
        )
        if sig is None or sig.direction == "flat":
            continue

        entry = sig.entry_price
        sl_dist = abs(entry - sig.stop_loss)
        tp_dist = abs(entry - sig.take_profit)
        if sl_dist <= 0 or entry <= 0:
            continue

        if sig.direction == "long":
            sl = entry - sl_dist
            tp = entry + tp_dist
        else:
            sl = entry + sl_dist
            tp = entry - tp_dist

        trail_active = False
        hwm = 0.0
        current_sl = sl
        exit_price = None
        exit_reason = None
        exit_idx = n - 1

        for j in range(i + 1, n):
            bh, bl = highs[j], lows[j]
            if sig.direction == "long":
                gain = (bh - entry) / entry
                if gain >= TRAIL_ACTIVATE:
                    trail_active = True
                if trail_active:
                    if bh > hwm:
                        hwm = bh
                    new_sl = hwm * (1 - TRAIL_OFFSET)
                    if new_sl > current_sl:
                        current_sl = new_sl
                sl_hit = bl <= current_sl
                tp_hit = bh >= tp
            else:
                gain = (entry - bl) / entry
                if gain >= TRAIL_ACTIVATE:
                    trail_active = True
                if trail_active:
                    if hwm == 0 or bl < hwm:
                        hwm = bl
                    new_sl = hwm * (1 + TRAIL_OFFSET)
                    if new_sl < current_sl:
                        current_sl = new_sl
                sl_hit = bh >= current_sl
                tp_hit = bl <= tp

            if sl_hit:
                exit_price = current_sl
                exit_reason = "trail" if trail_active else "sl"
                exit_idx = j
                break
            elif tp_hit:
                exit_price = tp
                exit_reason = "tp"
                exit_idx = j
                break

        if exit_price is None:
            exit_price = closes[-1]
            exit_reason = "timeout"
            exit_idx = n - 1

        if sig.direction == "long":
            raw_pct = (exit_price - entry) / entry
        else:
            raw_pct = (entry - exit_price) / entry

        last_exit_idx = exit_idx
        trades.append(PrecomputedTrade(
            symbol=symbol, tf=tf, direction=sig.direction,
            entry_time=pd.Timestamp(timestamps[i]),
            exit_time=pd.Timestamp(timestamps[exit_idx]),
            entry_price=entry, exit_price=exit_price,
            sl_distance=sl_dist, tp_distance=tp_dist,
            raw_pct=raw_pct, exit_reason=exit_reason,
            bars_held=exit_idx - i,
            ema50=float(ema50_vals[i]),
            ema200=float(ema200_vals[i]),
        ))

    return trades


def classify_trend(ema50: float, ema200: float) -> str:
    if np.isnan(ema50) or np.isnan(ema200) or ema200 == 0:
        return "neutral"
    spread = (ema50 - ema200) / ema200
    if spread > EMA_NEUTRAL_BAND:
        return "bull"
    elif spread < -EMA_NEUTRAL_BAND:
        return "bear"
    return "neutral"


def get_risk_for_trade(direction: str, trend: str,
                       with_risk: float, against_risk: float,
                       neutral_risk: float) -> float:
    if trend == "neutral":
        return neutral_risk
    if (direction == "long" and trend == "bull") or \
       (direction == "short" and trend == "bear"):
        return with_risk
    return against_risk


def run_portfolio(all_trades: list[PrecomputedTrade],
                  with_risk: float, against_risk: float,
                  neutral_risk: float) -> dict:
    equity = START_EQUITY
    peak = equity
    max_dd_pct = 0.0
    max_dd_usd = 0.0
    min_equity = equity

    events = []
    for idx, t in enumerate(all_trades):
        events.append(("entry", t.entry_time, idx))
        events.append(("exit", t.exit_time, idx))
    events.sort(key=lambda e: (e[1], 0 if e[0] == "exit" else 1))

    open_positions: dict[int, dict] = {}
    closed = wins = 0
    total_pnl = total_fees = 0.0
    open_symbols: set[str] = set()
    max_conc = skipped = 0
    tp_c = sl_c = trail_c = 0
    with_count = against_count = neutral_count = skip_count = 0

    for event_type, _, trade_idx in events:
        trade = all_trades[trade_idx]

        if event_type == "exit" and trade_idx in open_positions:
            pos = open_positions.pop(trade_idx)
            notional = pos["notional"]
            exit_fee = notional * TAKER_FEE
            pnl = notional * trade.raw_pct - pos["entry_fee"] - exit_fee
            equity += pnl
            total_pnl += pnl
            total_fees += pos["entry_fee"] + exit_fee
            closed += 1
            if pnl > 0:
                wins += 1
            if trade.exit_reason == "tp":
                tp_c += 1
            elif trade.exit_reason == "trail":
                trail_c += 1
            elif trade.exit_reason == "sl":
                sl_c += 1
            open_symbols.discard(f"{trade.symbol}_{trade.tf}")
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd_pct:
                max_dd_pct = dd
                max_dd_usd = peak - equity
            if equity < min_equity:
                min_equity = equity

        elif event_type == "entry":
            sym_key = f"{trade.symbol}_{trade.tf}"
            if (len(open_positions) >= MAX_CONCURRENT or
                    sym_key in open_symbols or equity <= 0):
                skipped += 1
                continue

            trend = classify_trend(trade.ema50, trade.ema200)
            risk_pct = get_risk_for_trade(
                trade.direction, trend, with_risk, against_risk, neutral_risk)

            if risk_pct <= 0:
                skip_count += 1
                skipped += 1
                continue

            if trend == "neutral":
                neutral_count += 1
            elif (trade.direction == "long" and trend == "bull") or \
                 (trade.direction == "short" and trend == "bear"):
                with_count += 1
            else:
                against_count += 1

            risk_amount = equity * risk_pct
            qty = risk_amount / (trade.sl_distance +
                                 trade.entry_price * ROUND_TRIP_COST)
            notional = qty * trade.entry_price
            max_notional = equity * LEVERAGE * NOTIONAL_CAP_FRAC
            if notional > max_notional:
                notional = max_notional

            open_positions[trade_idx] = {
                "notional": notional,
                "entry_fee": notional * MAKER_FEE,
            }
            open_symbols.add(sym_key)
            if len(open_positions) > max_conc:
                max_conc = len(open_positions)

    for trade_idx, pos in list(open_positions.items()):
        trade = all_trades[trade_idx]
        notional = pos["notional"]
        exit_fee = notional * TAKER_FEE
        pnl = notional * trade.raw_pct - pos["entry_fee"] - exit_fee
        equity += pnl
        total_pnl += pnl
        closed += 1
        if pnl > 0:
            wins += 1

    if equity > peak:
        peak = equity
    dd = (peak - equity) / peak if peak > 0 else 0
    if dd > max_dd_pct:
        max_dd_pct = dd
    if equity < min_equity:
        min_equity = equity

    wr = wins / closed * 100 if closed else 0
    return {
        "trades": closed, "wins": wins, "wr": wr,
        "pnl": total_pnl, "final_eq": equity,
        "total_fees": total_fees,
        "max_dd_pct": max_dd_pct, "max_dd_usd": max_dd_usd,
        "min_equity": min_equity,
        "max_concurrent": max_conc, "skipped": skipped,
        "tp": tp_c, "sl": sl_c, "trail": trail_c,
        "with_trend": with_count, "against_trend": against_count,
        "neutral": neutral_count, "skipped_against": skip_count,
    }


def load_candles(symbol, cache_dir):
    cache_path = cache_dir / f"{symbol}_{FROM}_{TO}_5m.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    return pd.DataFrame()


def main():
    cache_dir = project_root / "data_cache"

    print("=" * 95, flush=True)
    print("EMA TREND-ALIGNED SIZING SIMULATION", flush=True)
    print(f"EMA 50 vs EMA 200 on entry bar  |  Neutral band: +/-{EMA_NEUTRAL_BAND:.1%}",
          flush=True)
    print(f"Equity: ${START_EQUITY:,.0f}  |  Leverage: {LEVERAGE}x  |  "
          f"Max concurrent: {MAX_CONCURRENT}", flush=True)
    print("=" * 95, flush=True)

    # Collect trades once
    print("\n-- Collecting trades --", flush=True)
    all_trades: list[PrecomputedTrade] = []

    for sym in SYMBOLS:
        df_5m = load_candles(sym, cache_dir)
        if df_5m.empty:
            continue
        for tf in TIMEFRAMES:
            t0 = time.time()
            context_tf = CONTEXT_TF.get(tf, "4h")
            context_df = build_bars_for_tf(df_5m, context_tf)
            trading_bars = build_bars_for_tf(df_5m, tf)
            trades = collect_and_resolve(trading_bars, context_df, sym, tf)
            all_trades.extend(trades)
            print(f"  {tf} {sym}: {len(trades)} trades  ({time.time()-t0:.1f}s)",
                  flush=True)

    all_trades.sort(key=lambda t: t.entry_time)
    print(f"\n  Total: {len(all_trades)} trades", flush=True)

    # Trend breakdown of all signals
    bull_long = sum(1 for t in all_trades
                    if classify_trend(t.ema50, t.ema200) == "bull"
                    and t.direction == "long")
    bull_short = sum(1 for t in all_trades
                     if classify_trend(t.ema50, t.ema200) == "bull"
                     and t.direction == "short")
    bear_long = sum(1 for t in all_trades
                    if classify_trend(t.ema50, t.ema200) == "bear"
                    and t.direction == "long")
    bear_short = sum(1 for t in all_trades
                     if classify_trend(t.ema50, t.ema200) == "bear"
                     and t.direction == "short")
    neutral = sum(1 for t in all_trades
                  if classify_trend(t.ema50, t.ema200) == "neutral")

    print(f"\n  Signal breakdown:", flush=True)
    print(f"    WITH trend:    {bull_long + bear_short:>5d}  "
          f"(long in bull: {bull_long}, short in bear: {bear_short})", flush=True)
    print(f"    AGAINST trend: {bull_short + bear_long:>5d}  "
          f"(short in bull: {bull_short}, long in bear: {bear_long})", flush=True)
    print(f"    NEUTRAL:       {neutral:>5d}", flush=True)

    # Win rates by trend alignment
    with_wins = sum(1 for t in all_trades
                    if ((classify_trend(t.ema50, t.ema200) == "bull" and t.direction == "long") or
                        (classify_trend(t.ema50, t.ema200) == "bear" and t.direction == "short"))
                    and t.raw_pct > 0)
    with_total = bull_long + bear_short
    against_wins = sum(1 for t in all_trades
                       if ((classify_trend(t.ema50, t.ema200) == "bull" and t.direction == "short") or
                           (classify_trend(t.ema50, t.ema200) == "bear" and t.direction == "long"))
                       and t.raw_pct > 0)
    against_total = bull_short + bear_long
    neutral_wins = sum(1 for t in all_trades
                       if classify_trend(t.ema50, t.ema200) == "neutral"
                       and t.raw_pct > 0)

    print(f"\n  Raw win rates (before portfolio sizing):", flush=True)
    print(f"    WITH trend:    {with_wins}/{with_total}  = "
          f"{with_wins/with_total*100:.1f}%" if with_total else "", flush=True)
    print(f"    AGAINST trend: {against_wins}/{against_total}  = "
          f"{against_wins/against_total*100:.1f}%" if against_total else "", flush=True)
    print(f"    NEUTRAL:       {neutral_wins}/{neutral}  = "
          f"{neutral_wins/neutral*100:.1f}%" if neutral else "", flush=True)

    # Run each scenario
    print(f"\n{'=' * 95}", flush=True)
    print("RESULTS", flush=True)
    print("=" * 95, flush=True)

    results = []
    for label, wr, ar, nr in SCENARIOS:
        print(f"  Running: {label} ...", end=" ", flush=True)
        t0 = time.time()
        r = run_portfolio(all_trades, wr, ar, nr)
        r["label"] = label
        results.append(r)
        print(f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'Scenario':<32s}  {'Trades':>7s}  {'WR':>6s}  "
          f"{'Final Eq':>10s}  {'PnL':>12s}  {'Return':>8s}  "
          f"{'Max DD':>7s}  {'Min Eq':>9s}  {'With':>5s}  "
          f"{'Agst':>5s}  {'Skip':>5s}", flush=True)
    print("-" * 130, flush=True)

    for r in results:
        ret = r["pnl"] / START_EQUITY * 100
        print(f"{r['label']:<32s}  {r['trades']:>7d}  {r['wr']:>5.1f}%  "
              f"${r['final_eq']:>8,.0f}  ${r['pnl']:>+10,.0f}  "
              f"{ret:>+7.1f}%  {r['max_dd_pct']:>6.1%}  "
              f"${r['min_equity']:>7,.0f}  "
              f"{r['with_trend']:>5d}  {r['against_trend']:>5d}  "
              f"{r['skipped_against']:>5d}", flush=True)

    best = max(results, key=lambda r: r["final_eq"])
    safest = min(results, key=lambda r: r["max_dd_pct"])
    print(f"\n  BEST PnL:     {best['label']}  ->  ${best['final_eq']:,.0f}  "
          f"(DD {best['max_dd_pct']:.1%})", flush=True)
    print(f"  SAFEST:       {safest['label']}  ->  ${safest['final_eq']:,.0f}  "
          f"(DD {safest['max_dd_pct']:.1%})", flush=True)
    print("=" * 95, flush=True)


if __name__ == "__main__":
    main()
