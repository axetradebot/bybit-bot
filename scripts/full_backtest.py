"""
Full 3-year backtest of the Sniper strategy matching LIVE risk management.

Matches the production setup exactly:
  - Fee-adjusted position sizing (maker + taker + slippage in denominator)
  - 15% notional cap (equity * leverage * 0.15)
  - 20x leverage
  - 2% risk per trade
  - Trailing stop: activate at 10%, trail 3% behind HWM
  - Slippage on limit fills (0.03 bps adverse)
  - Compounding equity
  - Funding rate approximation (flat 0.01% per 8h)

Usage:
    python scripts/full_backtest.py
    python scripts/full_backtest.py --symbols SOLUSDT DOGEUSDT
    python scripts/full_backtest.py --equity 5000 --risk 0.02
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.indicators.resample import CONTEXT_TF, build_bars_for_tf
from src.strategies.strategy_sniper import SniperStrategy
from src.strategies.base import _sf

# ── Constants matching live risk_manager.py ──────────────────────────
MAKER_FEE = 0.0002
TAKER_FEE = 0.00055
SLIPPAGE_BPS = 0.0003
ROUND_TRIP_COST = MAKER_FEE + TAKER_FEE + SLIPPAGE_BPS
NOTIONAL_CAP_FRAC = 0.25
COOLDOWN_BARS = 6
FUNDING_RATE_8H = 0.0001      # approximate average funding
LIMIT_ORDER_TTL = 3

DEFAULT_SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT",
    "1000PEPEUSDT", "DOGEUSDT", "OPUSDT", "SUIUSDT",
]
TIMEFRAMES = ["15m", "4h"]


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class Trade:
    symbol: str
    tf: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    sl: float
    tp: float
    notional: float
    pnl_usd: float
    fees_usd: float
    funding_usd: float
    exit_reason: str
    bars_held: int
    equity_after: float


@dataclass
class Position:
    direction: str
    entry_price: float
    sl: float
    tp: float
    original_sl: float
    notional: float
    entry_time: datetime
    entry_bar_idx: int
    fee_entry: float
    trail_active: bool = False
    hwm: float = 0.0
    bars_held: int = 0
    funding_paid: float = 0.0


# ── Position sizing matching live risk_manager ───────────────────────

def compute_position_size(
    equity: float,
    risk_pct: float,
    entry_price: float,
    sl_distance: float,
    leverage: int,
) -> float:
    """Return notional USD, matching live _position_size_gate."""
    if sl_distance <= 0 or entry_price <= 0:
        return 0.0

    risk_amount = equity * risk_pct
    qty = risk_amount / (sl_distance + entry_price * ROUND_TRIP_COST)
    notional = qty * entry_price

    max_notional = equity * leverage * NOTIONAL_CAP_FRAC
    if notional > max_notional:
        notional = max_notional

    return notional


# ── Fetch / cache 5m candles ─────────────────────────────────────────

def fetch_5m_candles(symbol: str, since: str, until: str) -> pd.DataFrame:
    import ccxt
    exchange = ccxt.bybit({"enableRateLimit": True})
    exchange.load_markets()
    from src.live.order_manager import _to_ccxt_symbol
    bybit_symbol = _to_ccxt_symbol(symbol)
    if bybit_symbol not in exchange.markets:
        bybit_symbol = symbol.replace("USDT", "/USDT:USDT")
    since_ts = int(datetime.fromisoformat(since).replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    until_ts = int(datetime.fromisoformat(until).replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    all_candles: list[list] = []
    current = since_ts
    while current < until_ts:
        try:
            ohlcv = exchange.fetch_ohlcv(
                bybit_symbol, "5m", since=current, limit=1000)
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
    df = pd.DataFrame(all_candles,
                      columns=["timestamp", "open", "high", "low",
                               "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (df.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp").reset_index(drop=True))
    df["symbol"] = symbol
    df["buy_volume"] = df["volume"] * 0.5
    df["sell_volume"] = df["volume"] * 0.5
    df["volume_delta"] = 0.0
    df["quote_volume"] = df["volume"] * df["close"]
    df["trade_count"] = 0
    df["mark_price"] = df["close"]
    df["funding_rate"] = 0.0
    return df


def load_candles(symbol: str, from_date: str, to_date: str,
                 cache_dir: Path) -> pd.DataFrame:
    cache_path = cache_dir / f"{symbol}_{from_date}_{to_date}_5m.parquet"
    if cache_path.exists():
        print(f"  {symbol}: loading cache ...", flush=True)
        return pd.read_parquet(cache_path)
    print(f"  {symbol}: downloading ...", flush=True)
    df = fetch_5m_candles(symbol, from_date, to_date)
    if not df.empty:
        df.to_parquet(cache_path)
    return df


# ── Simulation engine ────────────────────────────────────────────────

def simulate_symbol_tf(
    symbol: str,
    tf: str,
    trading_bars: pd.DataFrame,
    context_df: pd.DataFrame | None,
    equity: float,
    risk_pct: float,
    leverage: int,
    trail_activate: float,
    trail_offset: float,
) -> tuple[list[Trade], float]:
    """
    Bar-by-bar simulation for one (symbol, tf) pair.
    Returns (trades, final_equity).
    """
    sniper = SniperStrategy()
    trades: list[Trade] = []

    ctx_ts: list[pd.Timestamp] = []
    ctx_rows: list = []
    if context_df is not None and not context_df.empty:
        for _, row in context_df.iterrows():
            ctx_ts.append(pd.Timestamp(row["timestamp"]))
            ctx_rows.append(row)

    def get_ctx(ts: pd.Timestamp):
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
    n_bars = len(trading_bars)

    pos: Position | None = None
    last_exit_idx = -COOLDOWN_BARS - 1
    pending_signal = None
    pending_idx = -1
    pending_bars = 0

    funding_interval = 96  # every 8h = 96 five-minute bars

    for i in range(n_bars):
        bar = trading_bars.iloc[i]
        bar_ts = pd.Timestamp(bar["timestamp"])
        bar_high = highs[i]
        bar_low = lows[i]
        bar_close = closes[i]

        # ── 1. Fill pending limit order ──
        if pending_signal is not None and pos is None:
            pending_bars += 1
            sig = pending_signal
            filled = False
            fill_price = 0.0

            if sig["direction"] == "long":
                if bar_low <= sig["limit_price"]:
                    fill_price = sig["limit_price"] * (1 + SLIPPAGE_BPS)
                    filled = True
            else:
                if bar_high >= sig["limit_price"]:
                    fill_price = sig["limit_price"] * (1 - SLIPPAGE_BPS)
                    filled = True

            if not filled and pending_bars >= LIMIT_ORDER_TTL:
                pending_signal = None

            if filled:
                sl_dist = sig["sl_distance"]
                tp_dist = sig["tp_distance"]

                if sig["direction"] == "long":
                    sl = fill_price - sl_dist
                    tp = fill_price + tp_dist
                else:
                    sl = fill_price + sl_dist
                    tp = fill_price - tp_dist

                notional = compute_position_size(
                    equity, risk_pct, fill_price, sl_dist, leverage)

                if notional <= 0:
                    pending_signal = None
                    continue

                entry_fee = notional * MAKER_FEE
                pos = Position(
                    direction=sig["direction"],
                    entry_price=fill_price,
                    sl=sl,
                    tp=tp,
                    original_sl=sl,
                    notional=notional,
                    entry_time=bar_ts.to_pydatetime(),
                    entry_bar_idx=i,
                    fee_entry=entry_fee,
                )

                # Same-bar SL check
                if sig["direction"] == "long" and bar_low <= sl:
                    exit_price = sl
                    raw_pct = (exit_price - fill_price) / fill_price
                    exit_fee = notional * TAKER_FEE
                    pnl = notional * raw_pct - entry_fee - exit_fee
                    equity += pnl
                    trades.append(Trade(
                        symbol=symbol, tf=tf, direction=sig["direction"],
                        entry_time=pos.entry_time, exit_time=bar_ts.to_pydatetime(),
                        entry_price=fill_price, exit_price=exit_price,
                        sl=sl, tp=tp, notional=notional,
                        pnl_usd=pnl, fees_usd=entry_fee + exit_fee,
                        funding_usd=0, exit_reason="sl",
                        bars_held=0, equity_after=equity,
                    ))
                    pos = None
                    last_exit_idx = i
                elif sig["direction"] == "short" and bar_high >= sl:
                    exit_price = sl
                    raw_pct = (fill_price - exit_price) / fill_price
                    exit_fee = notional * TAKER_FEE
                    pnl = notional * raw_pct - entry_fee - exit_fee
                    equity += pnl
                    trades.append(Trade(
                        symbol=symbol, tf=tf, direction=sig["direction"],
                        entry_time=pos.entry_time, exit_time=bar_ts.to_pydatetime(),
                        entry_price=fill_price, exit_price=exit_price,
                        sl=sl, tp=tp, notional=notional,
                        pnl_usd=pnl, fees_usd=entry_fee + exit_fee,
                        funding_usd=0, exit_reason="sl",
                        bars_held=0, equity_after=equity,
                    ))
                    pos = None
                    last_exit_idx = i

                pending_signal = None

        # ── 2. Manage open position ──
        if pos is not None and pos.entry_bar_idx < i:
            pos.bars_held += 1
            entry = pos.entry_price

            # Trailing stop
            if pos.direction == "long":
                gain = (bar_high - entry) / entry
                if gain >= trail_activate:
                    pos.trail_active = True
                if pos.trail_active:
                    if bar_high > pos.hwm:
                        pos.hwm = bar_high
                    new_sl = pos.hwm * (1 - trail_offset)
                    if new_sl > pos.sl:
                        pos.sl = new_sl
            else:
                gain = (entry - bar_low) / entry
                if gain >= trail_activate:
                    pos.trail_active = True
                if pos.trail_active:
                    if pos.hwm == 0 or bar_low < pos.hwm:
                        pos.hwm = bar_low
                    new_sl = pos.hwm * (1 + trail_offset)
                    if new_sl < pos.sl:
                        pos.sl = new_sl

            # Funding (approximate: charge every 96 bars = 8h)
            if pos.bars_held % funding_interval == 0:
                pos.funding_paid += pos.notional * FUNDING_RATE_8H

            # Check SL / TP
            exit_price = None
            exit_reason = None
            if pos.direction == "long":
                sl_hit = bar_low <= pos.sl
                tp_hit = bar_high >= pos.tp
            else:
                sl_hit = bar_high >= pos.sl
                tp_hit = bar_low <= pos.tp

            if sl_hit:
                exit_price = pos.sl
                exit_reason = "trail" if pos.trail_active else "sl"
            elif tp_hit:
                exit_price = pos.tp
                exit_reason = "tp"

            if exit_price is not None:
                if pos.direction == "long":
                    raw_pct = (exit_price - entry) / entry
                else:
                    raw_pct = (entry - exit_price) / entry

                exit_fee = pos.notional * TAKER_FEE
                total_fees = pos.fee_entry + exit_fee
                pnl = pos.notional * raw_pct - total_fees - pos.funding_paid
                equity += pnl

                trades.append(Trade(
                    symbol=symbol, tf=tf, direction=pos.direction,
                    entry_time=pos.entry_time, exit_time=bar_ts.to_pydatetime(),
                    entry_price=entry, exit_price=exit_price,
                    sl=pos.original_sl, tp=pos.tp, notional=pos.notional,
                    pnl_usd=pnl, fees_usd=total_fees,
                    funding_usd=pos.funding_paid, exit_reason=exit_reason,
                    bars_held=pos.bars_held, equity_after=equity,
                ))
                pos = None
                last_exit_idx = i

        # ── 3. Generate new signal ──
        if (pos is None and pending_signal is None
                and (i - last_exit_idx) >= COOLDOWN_BARS):
            ctx = get_ctx(bar_ts)
            sig = sniper.generate_signal(
                symbol=symbol, indicators_5m=bar,
                indicators_15m=ctx, funding_rate=0.0,
                liq_volume_1h=0.0)

            if sig is not None and sig.direction != "flat":
                sl_dist = abs(sig.entry_price - sig.stop_loss)
                tp_dist = abs(sig.entry_price - sig.take_profit)
                pending_signal = {
                    "direction": sig.direction,
                    "limit_price": sig.entry_price,
                    "sl_distance": sl_dist,
                    "tp_distance": tp_dist,
                }
                pending_idx = i
                pending_bars = 0

    # Force-close open position at end
    if pos is not None:
        exit_price = closes[-1]
        entry = pos.entry_price
        if pos.direction == "long":
            raw_pct = (exit_price - entry) / entry
        else:
            raw_pct = (entry - exit_price) / entry
        exit_fee = pos.notional * TAKER_FEE
        total_fees = pos.fee_entry + exit_fee
        pnl = pos.notional * raw_pct - total_fees - pos.funding_paid
        equity += pnl
        trades.append(Trade(
            symbol=symbol, tf=tf, direction=pos.direction,
            entry_time=pos.entry_time,
            exit_time=trading_bars.iloc[-1]["timestamp"],
            entry_price=entry, exit_price=exit_price,
            sl=pos.original_sl, tp=pos.tp, notional=pos.notional,
            pnl_usd=pnl, fees_usd=total_fees,
            funding_usd=pos.funding_paid, exit_reason="timeout",
            bars_held=pos.bars_held, equity_after=equity,
        ))

    return trades, equity


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--from-date", default="2023-01-01")
    parser.add_argument("--to-date", default="2026-04-03")
    parser.add_argument("--equity", type=float, default=3000.0)
    parser.add_argument("--risk", type=float, default=0.02)
    parser.add_argument("--leverage", type=int, default=20)
    parser.add_argument("--trail-activate", type=float, default=0.10)
    parser.add_argument("--trail-offset", type=float, default=0.03)
    args = parser.parse_args()

    cache_dir = project_root / "data_cache"
    cache_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("SNIPER STRATEGY — 3-YEAR BACKTEST (LIVE-MATCHED)")
    print("=" * 72)
    print(f"  Equity:       ${args.equity:,.0f}")
    print(f"  Risk/trade:   {args.risk:.1%}")
    print(f"  Leverage:     {args.leverage}x")
    print(f"  Notional cap: {NOTIONAL_CAP_FRAC:.0%} of max leverage")
    print(f"  Fees:         maker {MAKER_FEE:.2%} + taker {TAKER_FEE:.3%} "
          f"+ slippage {SLIPPAGE_BPS:.2%}")
    print(f"  Trail:        activate@{args.trail_activate:.0%} / "
          f"offset {args.trail_offset:.0%}")
    print(f"  Period:       {args.from_date} -> {args.to_date}")
    print(f"  Symbols:      {', '.join(args.symbols)}")
    print("=" * 72)

    # Load all candle data
    print("\n-- Loading candle data --", flush=True)
    candle_cache: dict[str, pd.DataFrame] = {}
    for sym in args.symbols:
        candle_cache[sym] = load_candles(
            sym, args.from_date, args.to_date, cache_dir)
        print(f"    {sym}: {len(candle_cache[sym]):,} bars", flush=True)

    # Precompute indicator DataFrames for all (symbol, tf) combinations
    print("\n-- Computing indicators --", flush=True)
    precomputed: dict[tuple[str, str], tuple[pd.DataFrame, pd.DataFrame]] = {}
    total_combos = len(TIMEFRAMES) * len(args.symbols)
    combo_idx = 0
    for tf in TIMEFRAMES:
        for sym in args.symbols:
            combo_idx += 1
            df_5m = candle_cache[sym]
            if df_5m.empty:
                continue
            t0 = time.time()
            context_tf = CONTEXT_TF.get(tf, "4h")
            context_df = build_bars_for_tf(df_5m, context_tf)
            trading_bars = build_bars_for_tf(df_5m, tf) if tf != "5m" else df_5m
            elapsed = time.time() - t0
            precomputed[(sym, tf)] = (trading_bars, context_df)
            print(f"  [{combo_idx}/{total_combos}] {tf} {sym}: "
                  f"{len(trading_bars):,} trading bars, "
                  f"{len(context_df):,} context bars  ({elapsed:.1f}s)",
                  flush=True)

    # Run each (symbol, tf) pair INDEPENDENTLY with starting equity,
    # then aggregate. This avoids sequential equity contamination.
    all_trades: list[Trade] = []
    sym_results: list[dict] = []

    print("\n-- Running simulations (independent per pair) --", flush=True)
    combo_idx = 0
    for tf in TIMEFRAMES:
        for sym in args.symbols:
            combo_idx += 1
            if (sym, tf) not in precomputed:
                continue
            trading_bars, context_df = precomputed[(sym, tf)]

            t0 = time.time()
            trades, final_eq = simulate_symbol_tf(
                symbol=sym, tf=tf,
                trading_bars=trading_bars,
                context_df=context_df,
                equity=args.equity,
                risk_pct=args.risk,
                leverage=args.leverage,
                trail_activate=args.trail_activate,
                trail_offset=args.trail_offset,
            )
            elapsed = time.time() - t0
            all_trades.extend(trades)

            pnl = final_eq - args.equity
            wins = sum(1 for t in trades if t.pnl_usd > 0)
            wr = wins / len(trades) * 100 if trades else 0
            ret_pct = pnl / args.equity * 100
            print(f"  [{combo_idx}/{total_combos}] {tf:>3s} {sym:<16s}  "
                  f"trades={len(trades):>4d}  WR={wr:5.1f}%  "
                  f"PnL=${pnl:>+10,.2f} ({ret_pct:>+6.1f}%)  "
                  f"final=${final_eq:>10,.2f}  ({elapsed:.1f}s)", flush=True)

            sym_results.append({
                "tf": tf, "symbol": sym, "trades": len(trades),
                "wins": wins, "wr": wr, "pnl": pnl,
                "final_eq": final_eq, "ret_pct": ret_pct,
            })

    equity = args.equity + sum(r["pnl"] for r in sym_results)

    # ── Summary ──
    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t.pnl_usd > 0)
    total_losses = total_trades - total_wins
    total_pnl = equity - args.equity
    total_fees = sum(t.fees_usd for t in all_trades)
    total_funding = sum(t.funding_usd for t in all_trades)
    win_rate = total_wins / total_trades * 100 if total_trades else 0

    win_pnls = [t.pnl_usd for t in all_trades if t.pnl_usd > 0]
    loss_pnls = [t.pnl_usd for t in all_trades if t.pnl_usd <= 0]
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    profit_factor = (
        abs(sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) != 0
        else float("inf")
    )

    # Max drawdown from aggregated PnL (sort all trades by exit time)
    all_trades_sorted = sorted(all_trades, key=lambda t: t.exit_time)
    running_eq = args.equity
    peak = running_eq
    max_dd = 0.0
    max_dd_usd = 0.0
    for t in all_trades_sorted:
        running_eq += t.pnl_usd
        if running_eq > peak:
            peak = running_eq
        dd = (peak - running_eq) / peak if peak > 0 else 0
        dd_usd = peak - running_eq
        if dd > max_dd:
            max_dd = dd
            max_dd_usd = dd_usd

    # Monthly breakdown
    monthly: dict[str, float] = {}
    for t in all_trades_sorted:
        key = t.exit_time.strftime("%Y-%m") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:7]
        monthly[key] = monthly.get(key, 0) + t.pnl_usd

    # Exit reason breakdown
    reasons: dict[str, int] = {}
    for t in all_trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    # Direction breakdown
    long_trades = [t for t in all_trades if t.direction == "long"]
    short_trades = [t for t in all_trades if t.direction == "short"]
    long_pnl = sum(t.pnl_usd for t in long_trades)
    short_pnl = sum(t.pnl_usd for t in short_trades)
    long_wr = (sum(1 for t in long_trades if t.pnl_usd > 0)
               / len(long_trades) * 100 if long_trades else 0)
    short_wr = (sum(1 for t in short_trades if t.pnl_usd > 0)
                / len(short_trades) * 100 if short_trades else 0)

    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    print(f"  Starting equity:    ${args.equity:>10,.2f}")
    print(f"  Final equity:       ${equity:>10,.2f}")
    print(f"  Total PnL:          ${total_pnl:>+10,.2f} "
          f"({total_pnl / args.equity:>+.1%})")
    print(f"  Total fees paid:    ${total_fees:>10,.2f}")
    print(f"  Total funding paid: ${total_funding:>10,.2f}")
    print()
    print(f"  Total trades:       {total_trades:>6d}")
    print(f"  Wins / Losses:      {total_wins:>6d} / {total_losses}")
    print(f"  Win rate:           {win_rate:>6.1f}%")
    print(f"  Avg win:            ${avg_win:>+10,.2f}")
    print(f"  Avg loss:           ${avg_loss:>+10,.2f}")
    print(f"  Profit factor:      {profit_factor:>10.2f}")
    print()
    print(f"  Max drawdown:       {max_dd:>6.1%}  (${max_dd_usd:>,.2f})")
    print()
    print(f"  Longs:  {len(long_trades):>4d} trades  "
          f"WR={long_wr:5.1f}%  PnL=${long_pnl:>+,.2f}")
    print(f"  Shorts: {len(short_trades):>4d} trades  "
          f"WR={short_wr:5.1f}%  PnL=${short_pnl:>+,.2f}")
    print()
    print("  Exit reasons:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        pnl_r = sum(t.pnl_usd for t in all_trades if t.exit_reason == reason)
        print(f"    {reason:<10s}: {count:>5d} trades  PnL=${pnl_r:>+,.2f}")

    print()
    print("  Fee / slippage breakdown:")
    print(f"    Maker (entry):     {MAKER_FEE:.2%} per side")
    print(f"    Taker (SL/TP/exit):{TAKER_FEE:.3%} per side")
    print(f"    Slippage:          {SLIPPAGE_BPS:.2%} on fills")
    print(f"    Funding (approx):  {FUNDING_RATE_8H:.2%} per 8h")

    # Monthly PnL table
    if monthly:
        print()
        print("  Monthly PnL:")
        sorted_months = sorted(monthly.keys())
        for m in sorted_months:
            bar = "+" * max(0, int(monthly[m] / 20)) if monthly[m] > 0 else \
                  "-" * max(0, int(abs(monthly[m]) / 20))
            print(f"    {m}:  ${monthly[m]:>+9,.2f}  {bar}")

    # Per-pair ranking
    print()
    print("  Per-pair ranking (best to worst):")
    print(f"  {'TF':>3s}  {'Symbol':<16s}  {'Trades':>6s}  {'WR':>6s}  "
          f"{'PnL':>12s}  {'Return':>8s}", flush=True)
    print("  " + "-" * 60)
    for r in sorted(sym_results, key=lambda x: x["pnl"], reverse=True):
        print(f"  {r['tf']:>3s}  {r['symbol']:<16s}  {r['trades']:>6d}  "
              f"{r['wr']:>5.1f}%  ${r['pnl']:>+10,.2f}  "
              f"{r['ret_pct']:>+6.1f}%", flush=True)

    profitable = [r for r in sym_results if r["pnl"] > 0]
    losing = [r for r in sym_results if r["pnl"] <= 0]
    print(f"\n  Profitable pairs: {len(profitable)}/{len(sym_results)}")

    # Annualized return
    years = 3.0
    if equity > 0 and args.equity > 0 and equity > args.equity:
        cagr = (equity / args.equity) ** (1 / years) - 1
        print(f"\n  CAGR:  {cagr:>+.1%}")
    else:
        print(f"\n  CAGR:  n/a (net loss)")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
