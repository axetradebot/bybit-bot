"""
Filter ablation study for the high_winrate strategy.

Phase 1: Scan all bars. For each StochRSI trigger, record per-filter
         scores (pass/fail for each of the 11 filters).
Phase 2: Replay trades with different filter masks to isolate the
         marginal contribution of each filter to win rate & PnL.

Tests:
  - Trigger only (no filters)             → baseline WR
  - All filters ON                        → max-filter WR
  - Each filter removed one at a time     → impact of that filter
  - Each filter alone (trigger + 1 filter)→ individual filter value
  - Score thresholds (50%, 60%, 70%, 80%, 90%) → min confirmation level
  - All 2^11 subsets (sampled)            → find optimal combination

Usage:
    python scripts/ablation_filters.py --symbol ETHUSDT
"""
from __future__ import annotations

import argparse
import bisect
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
    MAKER_FEE, TAKER_FEE, LIMIT_ORDER_TTL,
)
from src.strategies.base import _sf, _valid

log = structlog.get_logger()

FILTER_NAMES = [
    "ema_stack",       # 0
    "macd",            # 1
    "rsi_zone",        # 2
    "mfi",             # 3
    "supertrend",      # 4
    "vwap",            # 5
    "ofi",             # 6
    "heikin_ashi",     # 7
    "htf_15m",         # 8
    "no_divergence",   # 9
    "near_ema",        # 10
]
N_FILTERS = len(FILTER_NAMES)

TP_MULT = 1.2   # 1:1.2 R:R
SL_ATR_MULT = 3.0


@dataclass
class TriggerSignal:
    bar_idx: int
    direction: str
    entry_price: float
    sl_distance: float
    tp_distance: float
    ofi: float
    filter_scores: list[int]   # 1=pass, 0=fail, -1=veto for each filter
    fill_mode: str = "market"


# ── Data loading ─────────────────────────────────────────────────────────

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
        for col in ["open", "high", "low", "close", "volume",
                     "buy_volume", "sell_volume"]:
            if col in self.bars_df.columns:
                self.bars_df[col] = pd.to_numeric(self.bars_df[col], errors="coerce")

        self.highs = self.bars_df["high"].values.astype(np.float64)
        self.lows = self.bars_df["low"].values.astype(np.float64)
        self.opens = self.bars_df["open"].values.astype(np.float64)
        self.closes = self.bars_df["close"].values.astype(np.float64)
        self.n_bars = len(self.bars_df)

        df_15m = pd.read_sql(
            "SELECT * FROM indicators_15m "
            "WHERE symbol = %(symbol)s "
            "  AND timestamp >= %(from_date)s AND timestamp < %(to_date)s "
            "ORDER BY timestamp",
            engine,
            params={"symbol": symbol, "from_date": from_date, "to_date": to_date},
            parse_dates=["timestamp"],
        )
        self._15m_ts = [pd.Timestamp(r["timestamp"]) for _, r in df_15m.iterrows()]
        self._15m_rows = [r for _, r in df_15m.iterrows()]

        funding_df = pd.read_sql(
            "SELECT timestamp, funding_rate FROM funding_history "
            "WHERE symbol = %(symbol)s ORDER BY timestamp",
            engine, params={"symbol": symbol}, parse_dates=["timestamp"],
        )
        self.funding_by_bar: dict[int, float] = {}
        ts_set = {}
        for _, r in funding_df.iterrows():
            ts_set[pd.Timestamp(r["timestamp"])] = float(r["funding_rate"])
        ts_arr = self.bars_df["timestamp"].values
        for i in range(len(ts_arr)):
            pts = pd.Timestamp(ts_arr[i])
            if pts in ts_set:
                self.funding_by_bar[i] = ts_set[pts]

        elapsed = time.time() - t0
        print(f"  Data loaded: {self.n_bars:,} bars, "
              f"{len(df_15m)} 15m rows, {len(funding_df)} funding rows "
              f"({elapsed:.1f}s)", flush=True)


# ── Phase 1: generate triggers with per-filter scores ────────────────────

def scan_triggers(data: PreloadedData) -> list[TriggerSignal]:
    bars_df = data.bars_df
    ts_15m = data._15m_ts
    rows_15m = data._15m_rows
    n = len(bars_df)
    signals: list[TriggerSignal] = []

    prev_k: float | None = None
    prev_d: float | None = None

    for i in range(n):
        c = bars_df.iloc[i]

        stoch_k = _sf(c.get("stochrsi_k"))
        stoch_d = _sf(c.get("stochrsi_d"))
        pk, pd_ = prev_k, prev_d
        prev_k, prev_d = stoch_k, stoch_d

        if pk is None or pd_ is None:
            continue

        long_cross = pk <= pd_ and stoch_k > stoch_d
        short_cross = pk >= pd_ and stoch_k < stoch_d
        if not long_cross and not short_cross:
            continue

        direction = "long" if long_cross else "short"
        if direction == "long" and stoch_k > 85:
            continue
        if direction == "short" and stoch_k < 15:
            continue

        atr = _sf(c.get("atr_14"))
        close = _sf(c.get("close"))
        if atr <= 0 or close <= 0:
            continue

        required = ("ema_9", "ema_21", "ema_50", "rsi_14",
                    "macd_hist", "supertrend_dir", "vwap",
                    "order_flow_imb", "ha_close", "ha_open")
        if not all(_valid(c.get(k)) for k in required):
            continue

        ema9 = _sf(c.get("ema_9"))
        ema21 = _sf(c.get("ema_21"))
        ema50 = _sf(c.get("ema_50"))
        rsi = _sf(c.get("rsi_14"))
        mfi = _sf(c.get("mfi_14"))
        macd_h = _sf(c.get("macd_hist"))
        st_dir = _sf(c.get("supertrend_dir"))
        vwap = _sf(c.get("vwap"))
        ofi = _sf(c.get("order_flow_imb"))
        ha_close = _sf(c.get("ha_close"))
        ha_open = _sf(c.get("ha_open"))

        scores = [0] * N_FILTERS

        # 0: EMA stack
        if direction == "long":
            if ema9 > ema21 > ema50:
                scores[0] = 1
            elif ema9 < ema21 < ema50:
                scores[0] = -1  # veto
        else:
            if ema9 < ema21 < ema50:
                scores[0] = 1
            elif ema9 > ema21 > ema50:
                scores[0] = -1

        # 1: MACD
        if (direction == "long" and macd_h > 0) or \
           (direction == "short" and macd_h < 0):
            scores[1] = 1

        # 2: RSI zone
        if 35 <= rsi <= 65:
            scores[2] = 1

        # 3: MFI
        if _valid(c.get("mfi_14")):
            if (direction == "long" and mfi > 45) or \
               (direction == "short" and mfi < 55):
                scores[3] = 1
        else:
            scores[3] = -2  # unavailable, ignore

        # 4: Supertrend
        if (direction == "long" and st_dir > 0) or \
           (direction == "short" and st_dir < 0):
            scores[4] = 1

        # 5: VWAP
        if (direction == "long" and close > vwap) or \
           (direction == "short" and close < vwap):
            scores[5] = 1

        # 6: OFI
        if (direction == "long" and ofi > 0.55) or \
           (direction == "short" and ofi < 0.45):
            scores[6] = 1

        # 7: Heikin Ashi
        if (direction == "long" and ha_close > ha_open) or \
           (direction == "short" and ha_close < ha_open):
            scores[7] = 1

        # 8: 15m HTF
        if all(_valid(c.get(k) if hasattr(c, 'get') else None)
               for k in []) or True:
            ts = pd.Timestamp(c.get("timestamp"))
            idx_15 = bisect.bisect_right(ts_15m, ts) - 1
            if idx_15 >= 0 and ts_15m:
                c15 = rows_15m[idx_15]
                if all(_valid(c15.get(k)) for k in ("ema_9", "ema_21")):
                    e9_15 = _sf(c15.get("ema_9"))
                    e21_15 = _sf(c15.get("ema_21"))
                    if (direction == "long" and e9_15 > e21_15) or \
                       (direction == "short" and e9_15 < e21_15):
                        scores[8] = 1
                else:
                    scores[8] = -2
            else:
                scores[8] = -2

        # 9: No divergence
        extras = c.get("extras") or {}
        if isinstance(extras, dict):
            has_counter = False
            if direction == "long" and (extras.get("div_regular_bear") or
                                         extras.get("mom_wave_regular_bear")):
                has_counter = True
            if direction == "short" and (extras.get("div_regular_bull") or
                                          extras.get("mom_wave_regular_bull")):
                has_counter = True
            if has_counter:
                scores[9] = -1  # veto
            else:
                scores[9] = 1
        else:
            scores[9] = -2

        # 10: Near EMA21
        if abs(close - ema21) <= 2.0 * atr:
            scores[10] = 1

        sl_dist = SL_ATR_MULT * atr
        tp_dist = sl_dist * TP_MULT

        signals.append(TriggerSignal(
            bar_idx=i,
            direction=direction,
            entry_price=close,
            sl_distance=sl_dist,
            tp_distance=tp_dist,
            ofi=ofi,
            filter_scores=scores,
        ))

    return signals


# ── Phase 2: replay with a filter mask ───────────────────────────────────

@dataclass
class ReplayResult:
    label: str
    n_trades: int
    wins: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    max_dd: float


def replay_with_mask(
    signals: list[TriggerSignal],
    data: PreloadedData,
    required_filters: set[int],
    veto_filters: set[int],
    min_score_ratio: float = 0.0,
    cooldown: int = 6,
    tp_mult_override: float | None = None,
    equity_start: float = 10_000.0,
    leverage: int = 10,
    risk_pct: float = 0.02,
) -> ReplayResult | None:
    """
    Replay using only signals that pass the required filter mask.
    required_filters: set of filter indices that MUST pass (score=1)
    veto_filters: set of filter indices that block trade if score=-1
    min_score_ratio: min fraction of available filters that must pass
    """
    highs, lows, opens, closes = data.highs, data.lows, data.opens, data.closes
    n_bars = data.n_bars
    funding = data.funding_by_bar

    equity = equity_start
    pnls: list[float] = []
    wins_list: list[float] = []
    losses_list: list[float] = []
    last_exit_bar = -cooldown

    for sig in signals:
        if sig.bar_idx - last_exit_bar < cooldown:
            continue

        sc = sig.filter_scores

        # Check vetoes
        vetoed = False
        for f in veto_filters:
            if sc[f] == -1:
                vetoed = True
                break
        if vetoed:
            continue

        # Check required filters
        skip = False
        for f in required_filters:
            if sc[f] != 1:
                skip = True
                break
        if skip:
            continue

        # Score ratio check (across all available filters)
        if min_score_ratio > 0:
            available = sum(1 for s in sc if s != -2)
            passed = sum(1 for s in sc if s == 1)
            if available < 4 or (passed / available) < min_score_ratio:
                continue

        sl_dist = sig.sl_distance
        tp_dist = sig.tp_distance
        if tp_mult_override is not None:
            tp_dist = sl_dist * tp_mult_override

        # Market fill at next bar open
        fill_bar = sig.bar_idx + 1
        if fill_bar >= n_bars:
            continue
        fill_price = opens[fill_bar]

        if sig.direction == "long":
            sl_price = fill_price - sl_dist
            tp_price = fill_price + tp_dist
        else:
            sl_price = fill_price + sl_dist
            tp_price = fill_price - tp_dist

        sl_pct = sl_dist / fill_price if fill_price > 0 else 0.01
        risk_amount = equity * risk_pct
        pos_size = min(risk_amount / sl_pct, equity * leverage) if sl_pct > 0 else risk_amount
        entry_fee = pos_size * TAKER_FEE
        funding_paid = 0.0

        # Same-bar exit
        if (sig.direction == "long" and lows[fill_bar] <= sl_price) or \
           (sig.direction == "short" and highs[fill_bar] >= sl_price):
            raw = (sl_price - fill_price) / fill_price if sig.direction == "long" \
                  else (fill_price - sl_price) / fill_price
            pnl = pos_size * raw - (entry_fee + pos_size * TAKER_FEE)
            equity += pnl
            pnls.append(pnl)
            (wins_list if pnl > 0 else losses_list).append(pnl)
            last_exit_bar = fill_bar
            continue

        # Scan for exit
        exit_price = closes[n_bars - 1]
        exit_bar = n_bars - 1
        for b in range(fill_bar + 1, n_bars):
            fr = funding.get(b)
            if fr is not None:
                if sig.direction == "long":
                    funding_paid += pos_size * fr
                else:
                    funding_paid -= pos_size * fr

            if sig.direction == "long":
                sl_hit = lows[b] <= sl_price
                tp_hit = highs[b] >= tp_price
            else:
                sl_hit = highs[b] >= sl_price
                tp_hit = lows[b] <= tp_price

            if sl_hit or tp_hit:
                exit_price = sl_price if sl_hit else tp_price
                exit_bar = b
                break

        raw = (exit_price - fill_price) / fill_price if sig.direction == "long" \
              else (fill_price - exit_price) / fill_price
        exit_fee = pos_size * TAKER_FEE
        pnl = pos_size * raw - funding_paid - (entry_fee + exit_fee)
        equity += pnl
        pnls.append(pnl)
        (wins_list if pnl > 0 else losses_list).append(pnl)
        last_exit_bar = exit_bar

    if not pnls:
        return None

    total_pnl = sum(pnls)
    gp = sum(wins_list) if wins_list else 0.0
    gl = abs(sum(losses_list)) if losses_list else 0.0
    pf = gp / gl if gl > 1e-9 else (float('inf') if gp > 0 else 0.0)

    eq = equity_start
    peak = eq
    max_dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    n_wins = len(wins_list)
    return ReplayResult(
        label="",
        n_trades=len(pnls),
        wins=n_wins,
        win_rate=n_wins / len(pnls),
        total_pnl=total_pnl,
        profit_factor=pf,
        expectancy=float(np.mean(pnls)),
        avg_win=float(np.mean(wins_list)) if wins_list else 0.0,
        avg_loss=float(np.mean(losses_list)) if losses_list else 0.0,
        max_dd=max_dd,
    )


def print_result(r: ReplayResult):
    pf_s = f"{r.profit_factor:.2f}" if not math.isinf(r.profit_factor) else "inf"
    print(f"  {r.label:<40s}  n={r.n_trades:>4}  WR={r.win_rate:>5.1%}  "
          f"PnL={r.total_pnl:>+8,.0f}  PF={pf_s:>5}  "
          f"Exp={r.expectancy:>+6,.0f}  AvgW={r.avg_win:>+6,.0f}  "
          f"AvgL={r.avg_loss:>+6,.0f}  DD={r.max_dd:>6,.0f}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--from", dest="from_date",
                   default=(date.today() - timedelta(days=365)).isoformat())
    p.add_argument("--to", dest="to_date", default=date.today().isoformat())
    return p.parse_args()


def main():
    args = parse_args()
    engine = create_engine(settings.sync_db_url)

    print(f"\n{'='*90}", flush=True)
    print(f"  FILTER ABLATION STUDY -- {args.symbol}", flush=True)
    print(f"  Window: {args.from_date} -> {args.to_date}  |  R:R = 1:{TP_MULT}", flush=True)
    print(f"{'='*90}", flush=True)

    data = PreloadedData(engine, args.symbol, args.from_date, args.to_date)

    print(f"\n  Scanning triggers...", flush=True)
    t0 = time.time()
    triggers = scan_triggers(data)
    print(f"  => {len(triggers)} raw triggers in {time.time()-t0:.1f}s\n", flush=True)

    if not triggers:
        print("  No triggers found.", flush=True)
        return

    all_filters = set(range(N_FILTERS))
    veto_set = {0, 9}  # EMA stack and divergence can veto

    # ── Section 1: Baseline — trigger only (no filters) ──────────────
    print(f"{'-'*90}", flush=True)
    print(f"  SECTION 1: BASELINE", flush=True)
    print(f"{'-'*90}", flush=True)

    r = replay_with_mask(triggers, data, set(), set(), cooldown=6)
    if r:
        r.label = "Trigger only (no filters)"
        print_result(r)

    r = replay_with_mask(triggers, data, set(), veto_set, cooldown=6)
    if r:
        r.label = "Trigger + vetoes only"
        print_result(r)

    r = replay_with_mask(triggers, data, all_filters - {3, 8, 9},
                         veto_set, cooldown=6)
    if r:
        r.label = "All filters ON (excl optional)"
        print_result(r)

    r = replay_with_mask(triggers, data, set(), veto_set,
                         min_score_ratio=0.80, cooldown=6)
    if r:
        r.label = "Score >= 80%"
        print_result(r)

    # ── Section 2: Score thresholds ──────────────────────────────────
    print(f"\n{'-'*90}", flush=True)
    print(f"  SECTION 2: SCORE THRESHOLDS (what % of filters must agree?)", flush=True)
    print(f"{'-'*90}", flush=True)

    for threshold in [0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]:
        r = replay_with_mask(triggers, data, set(), veto_set,
                             min_score_ratio=threshold, cooldown=6)
        if r:
            r.label = f"Score >= {threshold:.0%}"
            print_result(r)

    # ── Section 3: Each filter ALONE (trigger + 1 filter) ────────────
    print(f"\n{'-'*90}", flush=True)
    print(f"  SECTION 3: EACH FILTER ALONE (trigger + that 1 filter required)", flush=True)
    print(f"{'-'*90}", flush=True)

    for i, name in enumerate(FILTER_NAMES):
        r = replay_with_mask(triggers, data, {i}, veto_set, cooldown=6)
        if r:
            r.label = f"+ {name}"
            print_result(r)
        else:
            print(f"  + {name:<37s}  no trades", flush=True)

    # ── Section 4: Remove one filter at a time ───────────────────────
    print(f"\n{'-'*90}", flush=True)
    print(f"  SECTION 4: REMOVE ONE AT A TIME (from 80% threshold)", flush=True)
    print(f"  (Higher WR after removal = that filter was HURTING)", flush=True)
    print(f"{'-'*90}", flush=True)

    base_r = replay_with_mask(triggers, data, set(), veto_set,
                              min_score_ratio=0.80, cooldown=6)
    if base_r:
        base_r.label = "ALL filters (80% threshold)"
        print_result(base_r)
        print("  ---", flush=True)

    for i, name in enumerate(FILTER_NAMES):
        # "Remove" filter by treating it as always-pass
        modified_triggers = []
        for t in triggers:
            mt = TriggerSignal(
                bar_idx=t.bar_idx, direction=t.direction,
                entry_price=t.entry_price, sl_distance=t.sl_distance,
                tp_distance=t.tp_distance, ofi=t.ofi,
                filter_scores=list(t.filter_scores),
            )
            mt.filter_scores[i] = 1  # force pass
            modified_triggers.append(mt)

        r = replay_with_mask(modified_triggers, data, set(), veto_set,
                             min_score_ratio=0.80, cooldown=6)
        if r:
            wr_delta = r.win_rate - (base_r.win_rate if base_r else 0)
            pnl_delta = r.total_pnl - (base_r.total_pnl if base_r else 0)
            r.label = f"- {name} (WR{wr_delta:+.1%} PnL{pnl_delta:+,.0f})"
            print_result(r)

    # ── Section 5: Best filter combinations (top pairs, triples) ─────
    print(f"\n{'-'*90}", flush=True)
    print(f"  SECTION 5: BEST FILTER COMBINATIONS (required filters + vetoes)", flush=True)
    print(f"{'-'*90}", flush=True)

    # Test all pairs
    pair_results: list[tuple[ReplayResult, set]] = []
    for combo in itertools.combinations(range(N_FILTERS), 2):
        r = replay_with_mask(triggers, data, set(combo), veto_set, cooldown=6)
        if r and r.n_trades >= 10:
            names = "+".join(FILTER_NAMES[c] for c in combo)
            r.label = names
            pair_results.append((r, set(combo)))

    pair_results.sort(key=lambda x: x[0].total_pnl, reverse=True)
    print(f"\n  Top 10 pairs by PnL:", flush=True)
    for r, _ in pair_results[:10]:
        print_result(r)

    # Test all triples
    triple_results: list[tuple[ReplayResult, set]] = []
    for combo in itertools.combinations(range(N_FILTERS), 3):
        r = replay_with_mask(triggers, data, set(combo), veto_set, cooldown=6)
        if r and r.n_trades >= 10:
            names = "+".join(FILTER_NAMES[c] for c in combo)
            r.label = names
            triple_results.append((r, set(combo)))

    triple_results.sort(key=lambda x: x[0].total_pnl, reverse=True)
    print(f"\n  Top 10 triples by PnL:", flush=True)
    for r, _ in triple_results[:10]:
        print_result(r)

    # Test quads
    quad_results: list[tuple[ReplayResult, set]] = []
    for combo in itertools.combinations(range(N_FILTERS), 4):
        r = replay_with_mask(triggers, data, set(combo), veto_set, cooldown=6)
        if r and r.n_trades >= 10:
            names = "+".join(FILTER_NAMES[c] for c in combo)
            r.label = names
            quad_results.append((r, set(combo)))

    quad_results.sort(key=lambda x: x[0].total_pnl, reverse=True)
    print(f"\n  Top 10 quads by PnL:", flush=True)
    for r, _ in quad_results[:10]:
        print_result(r)

    # Test quints
    quint_results: list[tuple[ReplayResult, set]] = []
    for combo in itertools.combinations(range(N_FILTERS), 5):
        r = replay_with_mask(triggers, data, set(combo), veto_set, cooldown=6)
        if r and r.n_trades >= 10:
            names = "+".join(FILTER_NAMES[c] for c in combo)
            r.label = names
            quint_results.append((r, set(combo)))

    quint_results.sort(key=lambda x: x[0].total_pnl, reverse=True)
    print(f"\n  Top 10 quints by PnL:", flush=True)
    for r, _ in quint_results[:10]:
        print_result(r)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*90}", flush=True)
    print(f"  OVERALL BEST CONFIGURATIONS", flush=True)
    print(f"{'='*90}", flush=True)

    all_combos = pair_results + triple_results + quad_results + quint_results
    all_combos.sort(key=lambda x: x[0].total_pnl, reverse=True)

    print(f"\n  Top 15 across all sizes (min 10 trades):", flush=True)
    for r, fset in all_combos[:15]:
        print_result(r)

    # Best by WR (min 20 trades)
    wr_combos = [(r, f) for r, f in all_combos if r.n_trades >= 20]
    wr_combos.sort(key=lambda x: x[0].win_rate, reverse=True)
    print(f"\n  Top 10 by WIN RATE (min 20 trades):", flush=True)
    for r, fset in wr_combos[:10]:
        print_result(r)


if __name__ == "__main__":
    main()
