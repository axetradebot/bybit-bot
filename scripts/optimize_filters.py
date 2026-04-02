"""
Grid-search optimizer v3 — pre-generate signals, fast replay.

Phase 1: Scan all bars ONCE per strategy, record every raw signal.
Phase 2: For each param combo, replay signals with filters applied.
         Exit simulation only runs between entry → exit bars.

Usage:
    python scripts/optimize_filters.py --symbol BTCUSDT --from 2025-04-01 --to 2026-03-31
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import numpy as np
import pandas as pd
import structlog

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine
from src.config import settings
from src.backtest.simulator import (
    COOLDOWN_BARS, MAKER_FEE, TAKER_FEE, LIMIT_ORDER_TTL,
    classify_volatility, classify_funding, classify_time_of_day,
)
from src.strategies import STRATEGY_REGISTRY, LIMIT_FILL_HEADS, _sf

log = structlog.get_logger()

# ── Param grids ──────────────────────────────────────────────────────────

GRIDS: dict[str, dict[str, list]] = {
    "bb_squeeze": {
        "ofi_long":       [0.50, 0.52, 0.55, 0.57, 0.60, 0.63],
        "atr_rank_floor": [0.05, 0.10, 0.15, 0.20, 0.25],
        "tp_mult":        [1.5, 1.8, 2.0, 2.5, 3.0],
        "cooldown":       [4, 6, 8, 10, 12],
    },
    "multitf_scalp": {
        "ofi_long":       [0.50, 0.52, 0.55, 0.58, 0.62, 0.65],
        "atr_rank_floor": [0.05, 0.10, 0.15, 0.18, 0.22],
        "tp_mult":        [1.5, 2.0, 2.4, 2.8, 3.2],
        "cooldown":       [4, 6, 8, 10, 12],
    },
    "volume_delta_liq": {
        "ofi_long":       [0.50, 0.52, 0.55, 0.58, 0.60, 0.63],
        "atr_rank_floor": [0.05, 0.10, 0.15, 0.18, 0.22],
        "tp_mult":        [1.5, 1.8, 2.0, 2.5, 3.0],
        "cooldown":       [4, 6, 8, 10, 12],
    },
    "vwap_reversion": {
        "ofi_long":       [0.48, 0.50, 0.52, 0.55, 0.58, 0.60],
        "atr_rank_floor": [0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
        "tp_mult":        [1.2, 1.5, 1.8, 2.0, 2.2],
        "cooldown":       [4, 6, 8, 10, 12],
    },
    "rsi_divergence": {
        "tp_mult":        [1.5, 2.0, 2.5, 3.0, 3.5],
        "cooldown":       [4, 6, 8, 10, 12],
    },
    "regime_adaptive": {
        "tp_mult":        [1.5, 2.0, 2.5, 3.0],
        "cooldown":       [4, 6, 8, 10, 12],
    },
    "high_winrate": {
        "ofi_long":       [0.50],
        "atr_rank_floor": [0.10, 0.15, 0.20, 0.25, 0.30],
        "tp_mult":        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "cooldown":       [4, 6, 8, 12],
        "trailing":       [False],
    },
}


# ── Pre-loaded market data ───────────────────────────────────────────────

@dataclass
class BarData:
    """Flat arrays for ultra-fast exit simulation."""
    timestamp: np.ndarray       # datetime64
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    n_bars: int = 0


@dataclass
class RawSignal:
    """One raw signal produced by a strategy."""
    bar_idx: int
    direction: str              # "long" | "short"
    entry_price: float
    sl_distance: float
    tp_distance: float          # raw, before tp_mult override
    ofi: float                  # order_flow_imb value at signal bar
    atr_rank: float             # atr_pct_rank at signal bar
    fill_mode: str              # "limit" | "market"
    strategy_combo: list
    indicators_snapshot: dict


class PreloadedData:
    def __init__(self, engine, symbol: str, from_date: str, to_date: str):
        t0 = time.time()
        self.symbol = symbol

        candles = pd.read_sql(
            "SELECT * FROM candles_5m "
            "WHERE symbol = %(symbol)s "
            "  AND timestamp >= %(from_date)s AND timestamp < %(to_date)s "
            "ORDER BY timestamp",
            engine,
            params={"symbol": symbol, "from_date": from_date, "to_date": to_date},
            parse_dates=["timestamp"],
        )
        indicators = pd.read_sql(
            "SELECT * FROM indicators_5m "
            "WHERE symbol = %(symbol)s "
            "  AND timestamp >= %(from_date)s AND timestamp < %(to_date)s "
            "ORDER BY timestamp",
            engine,
            params={"symbol": symbol, "from_date": from_date, "to_date": to_date},
            parse_dates=["timestamp"],
        )
        self.bars_df = candles.merge(
            indicators, on=["symbol", "timestamp"],
            how="inner", suffixes=("", "_ind"),
        )
        numeric_cols = ["open", "high", "low", "close", "volume",
                        "buy_volume", "sell_volume"]
        for col in numeric_cols:
            if col in self.bars_df.columns:
                self.bars_df[col] = pd.to_numeric(self.bars_df[col], errors="coerce")

        self.bar_data = BarData(
            timestamp=self.bars_df["timestamp"].values,
            open=self.bars_df["open"].values.astype(np.float64),
            high=self.bars_df["high"].values.astype(np.float64),
            low=self.bars_df["low"].values.astype(np.float64),
            close=self.bars_df["close"].values.astype(np.float64),
            n_bars=len(self.bars_df),
        )

        df_15m = pd.read_sql(
            "SELECT * FROM indicators_15m "
            "WHERE symbol = %(symbol)s "
            "  AND timestamp >= %(from_date)s AND timestamp < %(to_date)s "
            "ORDER BY timestamp",
            engine,
            params={"symbol": symbol, "from_date": from_date, "to_date": to_date},
            parse_dates=["timestamp"],
        )
        self._15m_ts: list[pd.Timestamp] = []
        self._15m_rows: list[pd.Series] = []
        for _, row in df_15m.iterrows():
            self._15m_ts.append(pd.Timestamp(row["timestamp"]))
            self._15m_rows.append(row)

        funding_df = pd.read_sql(
            "SELECT timestamp, funding_rate FROM funding_history "
            "WHERE symbol = %(symbol)s ORDER BY timestamp",
            engine,
            params={"symbol": symbol},
            parse_dates=["timestamp"],
        )
        self.funding_map = {}
        ts_arr = self.bar_data.timestamp
        funding_ts_set = set()
        for _, row in funding_df.iterrows():
            pts = pd.Timestamp(row["timestamp"])
            self.funding_map[pts] = float(row["funding_rate"])
            funding_ts_set.add(pts)
        self.funding_bar_indices: list[tuple[int, float]] = []
        for i in range(len(ts_arr)):
            pts = pd.Timestamp(ts_arr[i])
            if pts in funding_ts_set:
                self.funding_bar_indices.append((i, self.funding_map[pts]))

        elapsed = time.time() - t0
        log.info("data_preloaded",
                 symbol=symbol, bars_5m=len(self.bars_df),
                 rows_15m=len(df_15m), funding_rows=len(funding_df),
                 seconds=f"{elapsed:.1f}")


# ── Phase 1: Generate all raw signals per strategy ───────────────────────

def generate_all_signals(
    strat_name: str,
    preloaded: PreloadedData,
) -> list[RawSignal]:
    """Call strategy.on_bar() for every bar and record raw signals."""
    import bisect

    base_cls = STRATEGY_REGISTRY[strat_name]
    base = base_cls()

    bars_df = preloaded.bars_df
    ts_15m = preloaded._15m_ts
    rows_15m = preloaded._15m_rows
    n = len(bars_df)
    signals: list[RawSignal] = []

    for i in range(n):
        bar = bars_df.iloc[i]
        prev_bar = bars_df.iloc[i - 1] if i > 0 else None

        ts = pd.Timestamp(bar["timestamp"])
        idx_15 = bisect.bisect_right(ts_15m, ts) - 1
        ind_15m = rows_15m[idx_15] if idx_15 >= 0 and ts_15m else pd.Series(dtype=object)

        funding = _sf(bar.get("funding_8h"))
        liq = _sf(bar.get("liq_volume_1h"))

        signal = base.generate_signal(
            symbol=str(bar.get("symbol", preloaded.symbol)),
            indicators_5m=bar,
            indicators_15m=ind_15m,
            funding_rate=funding,
            liq_volume_1h=liq,
        )

        if signal is None or signal.direction == "flat":
            continue

        fill_mode = signal.fill_mode
        if fill_mode is None:
            head = signal.strategy_combo[0] if signal.strategy_combo else ""
            fill_mode = "limit" if head in LIMIT_FILL_HEADS else "market"

        sl_dist = abs(signal.entry_price - signal.stop_loss)
        tp_dist = abs(signal.take_profit - signal.entry_price)
        ofi_val = _sf(bar.get("order_flow_imb"))
        atr_rank = _sf(bar.get("atr_pct_rank"))

        signals.append(RawSignal(
            bar_idx=i,
            direction=signal.direction,
            entry_price=float(signal.entry_price),
            sl_distance=sl_dist,
            tp_distance=tp_dist,
            ofi=ofi_val,
            atr_rank=atr_rank,
            fill_mode=fill_mode,
            strategy_combo=signal.strategy_combo,
            indicators_snapshot=signal.indicators_snapshot,
        ))

    return signals


# ── Phase 2: Ultra-fast replay ───────────────────────────────────────────

@dataclass
class RunResult:
    strategy: str
    params: dict
    n_trades: int
    total_pnl: float
    profit_factor: float
    expectancy: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_dd_usd: float
    max_dd_pct: float
    fees: float


def fast_replay(
    signals: list[RawSignal],
    bar_data: BarData,
    funding_indices: list[tuple[int, float]],
    params: dict,
    equity_start: float = 10_000.0,
    leverage: int = 10,
    risk_pct: float = 0.02,
    fixed_risk: bool = False,
) -> RunResult | None:
    """
    Replay pre-generated signals with param overrides.
    Only iterates bars between entry → exit (not all bars).
    """
    cooldown = params.get("cooldown", COOLDOWN_BARS)
    ofi_long = params.get("ofi_long")
    ofi_short = (1.0 - ofi_long) if ofi_long is not None else None
    atr_floor = params.get("atr_rank_floor")
    tp_mult = params.get("tp_mult")

    highs = bar_data.high
    lows = bar_data.low
    opens = bar_data.open
    closes = bar_data.close
    n_bars = bar_data.n_bars

    equity = equity_start
    pnls: list[float] = []
    fees_total = 0.0
    wins = 0
    losses_list: list[float] = []
    wins_list: list[float] = []
    last_exit_bar = -cooldown

    # Pre-index funding by bar
    funding_by_bar: dict[int, float] = dict(funding_indices)

    for sig in signals:
        if sig.bar_idx - last_exit_bar < cooldown:
            continue

        # ATR rank filter
        if atr_floor is not None and sig.atr_rank < atr_floor:
            continue

        # OFI directional filter
        if ofi_long is not None:
            if sig.direction == "long" and sig.ofi < ofi_long:
                continue
            if sig.direction == "short" and sig.ofi > ofi_short:
                continue

        # Apply TP multiplier
        sl_dist = sig.sl_distance
        if sl_dist < 1e-12:
            continue
        tp_dist = sl_dist * tp_mult if tp_mult is not None else sig.tp_distance

        # Min R:R check (1.0 minimum always)
        rr = tp_dist / sl_dist if sl_dist > 1e-9 else 0.0
        if rr < 1.0:
            continue

        fee_rate = TAKER_FEE if sig.fill_mode == "market" else MAKER_FEE

        # Determine fill
        if sig.fill_mode == "market":
            fill_bar = sig.bar_idx + 1
            if fill_bar >= n_bars:
                continue
            fill_price = opens[fill_bar]
        else:
            limit_price = sig.entry_price
            filled = False
            fill_bar = -1
            for b in range(sig.bar_idx + 1,
                           min(sig.bar_idx + 1 + LIMIT_ORDER_TTL, n_bars)):
                if sig.direction == "long":
                    if lows[b] <= limit_price:
                        fill_price = limit_price
                        fill_bar = b
                        filled = True
                        break
                else:
                    if highs[b] >= limit_price:
                        fill_price = limit_price
                        fill_bar = b
                        filled = True
                        break
            if not filled:
                continue

        # Position setup
        if sig.direction == "long":
            sl_price = fill_price - sl_dist
            tp_price = fill_price + tp_dist
        else:
            sl_price = fill_price + sl_dist
            tp_price = fill_price - tp_dist

        sl_pct = sl_dist / fill_price if fill_price > 0 else 0.01
        base_eq = equity_start if fixed_risk else equity
        risk_amount = base_eq * risk_pct
        pos_size = risk_amount / sl_pct if sl_pct > 0 else risk_amount
        pos_size = min(pos_size, equity * leverage)
        entry_fee = pos_size * fee_rate
        funding_paid = 0.0

        # Same-bar exit check
        if sig.direction == "long" and lows[fill_bar] <= sl_price:
            raw_pct = (sl_price - fill_price) / fill_price
            exit_fee = pos_size * fee_rate
            pnl = pos_size * raw_pct - (entry_fee + exit_fee)
            equity += pnl
            pnls.append(pnl)
            fees_total += entry_fee + exit_fee
            if pnl > 0:
                wins += 1; wins_list.append(pnl)
            else:
                losses_list.append(pnl)
            last_exit_bar = fill_bar
            continue
        elif sig.direction == "short" and highs[fill_bar] >= sl_price:
            raw_pct = (fill_price - sl_price) / fill_price
            exit_fee = pos_size * fee_rate
            pnl = pos_size * raw_pct - (entry_fee + exit_fee)
            equity += pnl
            pnls.append(pnl)
            fees_total += entry_fee + exit_fee
            if pnl > 0:
                wins += 1; wins_list.append(pnl)
            else:
                losses_list.append(pnl)
            last_exit_bar = fill_bar
            continue

        # Scan subsequent bars for exit (with trailing stop)
        exit_price = 0.0
        exit_bar = n_bars - 1  # fallback: close at last bar
        use_trail = params.get("trailing", False)
        best_price = fill_price

        for b in range(fill_bar + 1, n_bars):
            fr = funding_by_bar.get(b)
            if fr is not None:
                if sig.direction == "long":
                    funding_paid += pos_size * fr
                else:
                    funding_paid -= pos_size * fr

            bh = highs[b]
            bl = lows[b]

            # Trailing stop: after 1× R profit, move SL to breakeven
            # then trail at original SL distance from best price
            if use_trail:
                if sig.direction == "long":
                    best_price = max(best_price, bh)
                    if best_price >= fill_price + sl_dist:
                        trail_sl = best_price - sl_dist
                        sl_price = max(sl_price, trail_sl)
                else:
                    best_price = min(best_price, bl)
                    if best_price <= fill_price - sl_dist:
                        trail_sl = best_price + sl_dist
                        sl_price = min(sl_price, trail_sl)

            if sig.direction == "long":
                sl_hit = bl <= sl_price
                tp_hit = bh >= tp_price
            else:
                sl_hit = bh >= sl_price
                tp_hit = bl <= tp_price

            if sl_hit or tp_hit:
                exit_price = sl_price if sl_hit else tp_price
                exit_bar = b
                break
        else:
            exit_price = closes[n_bars - 1]
            exit_bar = n_bars - 1

        if sig.direction == "long":
            raw_pct = (exit_price - fill_price) / fill_price
        else:
            raw_pct = (fill_price - exit_price) / fill_price

        exit_fee = pos_size * fee_rate
        total_fees = entry_fee + exit_fee
        pnl = pos_size * raw_pct - funding_paid - total_fees
        equity += pnl
        pnls.append(pnl)
        fees_total += total_fees
        if pnl > 0:
            wins += 1; wins_list.append(pnl)
        else:
            losses_list.append(pnl)
        last_exit_bar = exit_bar

    if not pnls:
        return None

    total_pnl = sum(pnls)
    gross_profit = sum(wins_list) if wins_list else 0.0
    gross_loss_abs = abs(sum(losses_list)) if losses_list else 0.0
    pf = gross_profit / gross_loss_abs if gross_loss_abs > 1e-9 else (
        float('inf') if gross_profit > 0 else 0.0)

    eq = equity_start
    peak = eq
    max_dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = max_dd / peak if peak > 1e-9 else 0.0

    return RunResult(
        strategy="",
        params={},
        n_trades=len(pnls),
        total_pnl=total_pnl,
        profit_factor=pf,
        expectancy=float(np.mean(pnls)),
        win_rate=wins / len(pnls),
        avg_win=float(np.mean(wins_list)) if wins_list else 0.0,
        avg_loss=float(np.mean(losses_list)) if losses_list else 0.0,
        max_dd_usd=max_dd,
        max_dd_pct=max_dd_pct,
        fees=fees_total,
    )


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Grid-search strategy filter optimizer")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--from", dest="from_date",
                   default=(date.today() - timedelta(days=365)).isoformat())
    p.add_argument("--to", dest="to_date", default=date.today().isoformat())
    p.add_argument("--equity", type=float, default=10_000.0)
    p.add_argument("--risk-pct", type=float, default=0.02, dest="risk_pct")
    p.add_argument("--fixed-risk", action="store_true", dest="fixed_risk")
    p.add_argument("--strategy", default=None,
                   help="Single strategy to optimize (default: all)")
    p.add_argument("--max-combos", type=int, default=500,
                   help="Max param combos per strategy")
    return p.parse_args()


def sample_grid(grid: dict[str, list], max_combos: int) -> list[dict]:
    keys = sorted(grid.keys())
    all_vals = [grid[k] for k in keys]
    full = list(itertools.product(*all_vals))
    if len(full) <= max_combos:
        combos = full
    else:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(full), size=max_combos, replace=False)
        combos = [full[i] for i in sorted(indices)]
    return [dict(zip(keys, vals)) for vals in combos]


def main():
    args = parse_args()
    engine = create_engine(settings.sync_db_url)

    strategies = ([args.strategy] if args.strategy
                  else list(GRIDS.keys()))

    print(f"\nPreloading data for {args.symbol} ({args.from_date} -> {args.to_date})...",
          flush=True)
    preloaded = PreloadedData(engine, args.symbol, args.from_date, args.to_date)

    all_results: list[RunResult] = []
    grand_total = 0

    for strat_name in strategies:
        grid = GRIDS.get(strat_name, {})
        if not grid:
            print(f"No grid for {strat_name}, skipping", flush=True)
            continue

        combos = sample_grid(grid, args.max_combos)
        grand_total += len(combos)

        print(f"\n{'='*70}", flush=True)
        print(f"  Strategy: {strat_name}", flush=True)
        print(f"  Phase 1: generating raw signals (scanning {preloaded.bar_data.n_bars:,} bars)...",
              flush=True)

        t0 = time.time()
        raw_signals = generate_all_signals(strat_name, preloaded)
        sig_time = time.time() - t0
        print(f"  => {len(raw_signals)} raw signals in {sig_time:.1f}s", flush=True)
        print(f"  Phase 2: replaying {len(combos)} param combos...", flush=True)

        strat_results: list[RunResult] = []
        t0 = time.time()

        for i, params in enumerate(combos):
            result = fast_replay(
                signals=raw_signals,
                bar_data=preloaded.bar_data,
                funding_indices=preloaded.funding_bar_indices,
                params=params,
                equity_start=args.equity,
                risk_pct=getattr(args, 'risk_pct', 0.02),
                fixed_risk=getattr(args, 'fixed_risk', False),
            )

            if result and result.n_trades > 0:
                result.strategy = strat_name
                result.params = params
                strat_results.append(result)
                all_results.append(result)

            if (i + 1) % 50 == 0 or i == len(combos) - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                with_trades = len(strat_results)
                print(f"    [{i+1:4d}/{len(combos)}]  "
                      f"{with_trades} with trades  "
                      f"({rate:.0f} combos/s, {elapsed:.1f}s elapsed)",
                      flush=True)

        replay_time = time.time() - t0

        if strat_results:
            strat_results.sort(key=lambda r: r.total_pnl, reverse=True)
            best = strat_results[0]
            worst = strat_results[-1]
            pf_s = f"{best.profit_factor:.2f}" if not math.isinf(best.profit_factor) else "inf"
            print(f"\n  Best:  PnL={best.total_pnl:>+10,.0f}  PF={pf_s}  "
                  f"n={best.n_trades}  WR={best.win_rate:.0%}  "
                  f"DD={best.max_dd_usd:,.0f}", flush=True)
            print(f"         {best.params}", flush=True)
            pf_w = f"{worst.profit_factor:.2f}" if not math.isinf(worst.profit_factor) else "inf"
            print(f"  Worst: PnL={worst.total_pnl:>+10,.0f}  PF={pf_w}  "
                  f"n={worst.n_trades}  WR={worst.win_rate:.0%}  "
                  f"DD={worst.max_dd_usd:,.0f}", flush=True)
            print(f"         {worst.params}", flush=True)
        else:
            print(f"  No combos produced trades.", flush=True)
        print(f"  Replay: {len(combos)} combos in {replay_time:.1f}s "
              f"({len(combos)/replay_time:.0f} combos/s)" if replay_time > 0 else "",
              flush=True)

    if not all_results:
        print("\nNo results with trades.", flush=True)
        return

    all_results.sort(key=lambda r: r.total_pnl, reverse=True)

    os.makedirs("results", exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/optimization_{args.symbol}_{ts}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "strategy", "total_pnl", "profit_factor",
                     "expectancy", "n_trades", "win_rate",
                     "avg_win", "avg_loss", "max_dd_usd", "max_dd_pct",
                     "fees", "params"])
        for rank, r in enumerate(all_results, 1):
            w.writerow([
                rank, r.strategy, f"{r.total_pnl:.2f}", f"{r.profit_factor:.2f}",
                f"{r.expectancy:.2f}", r.n_trades, f"{r.win_rate:.4f}",
                f"{r.avg_win:.2f}", f"{r.avg_loss:.2f}",
                f"{r.max_dd_usd:.2f}", f"{r.max_dd_pct:.4f}",
                f"{r.fees:.2f}", str(r.params),
            ])

    print(f"\n{'='*70}", flush=True)
    print(f"  Results saved to {csv_path}", flush=True)
    print(f"  Total combos tested: {grand_total}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  TOP 20 BY TOTAL PnL (all strategies)", flush=True)
    print(f"{'='*70}", flush=True)
    hdr = (f"{'Rank':>4} {'Strategy':<20} {'PnL':>10} {'PF':>6} {'Trades':>6} "
           f"{'WR':>5} {'Exp$':>8} {'AvgW':>8} {'AvgL':>8} {'MaxDD':>8}")
    print(hdr, flush=True)
    print("-" * len(hdr) + "-" * 40, flush=True)
    for rank, r in enumerate(all_results[:20], 1):
        pf_s = f"{r.profit_factor:.2f}" if not math.isinf(r.profit_factor) else "inf"
        print(f"{rank:4d} {r.strategy:<20} {r.total_pnl:>+10,.0f} {pf_s:>6} "
              f"{r.n_trades:>6} {r.win_rate:>4.0%} {r.expectancy:>+8,.0f} "
              f"{r.avg_win:>+8,.0f} {r.avg_loss:>+8,.0f} {r.max_dd_usd:>8,.0f}  "
              f"{r.params}", flush=True)

    print(f"\n  BOTTOM 5:", flush=True)
    for rank, r in enumerate(all_results[-5:], len(all_results) - 4):
        pf_s = f"{r.profit_factor:.2f}" if not math.isinf(r.profit_factor) else "inf"
        print(f"{rank:4d} {r.strategy:<20} {r.total_pnl:>+10,.0f} {pf_s:>6} "
              f"{r.n_trades:>6} {r.win_rate:>4.0%} {r.expectancy:>+8,.0f} "
              f"{r.avg_win:>+8,.0f} {r.avg_loss:>+8,.0f} {r.max_dd_usd:>8,.0f}  "
              f"{r.params}", flush=True)

    print(f"\n{'='*70}", flush=True)
    print(f"  BEST PARAMS PER STRATEGY", flush=True)
    print(f"{'='*70}", flush=True)
    seen = set()
    for r in all_results:
        if r.strategy not in seen:
            seen.add(r.strategy)
            pf_s = f"{r.profit_factor:.2f}" if not math.isinf(r.profit_factor) else "inf"
            print(f"  {r.strategy:<20} PnL={r.total_pnl:>+10,.0f}  PF={pf_s}  "
                  f"n={r.n_trades}  WR={r.win_rate:.0%}  DD={r.max_dd_usd:,.0f}",
                  flush=True)
            print(f"    Params: {r.params}", flush=True)


if __name__ == "__main__":
    main()
