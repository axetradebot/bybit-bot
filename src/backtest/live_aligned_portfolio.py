"""
Live-aligned portfolio simulation helpers.

Mirrors production behaviour documented in ``order_manager.py`` and
``risk_manager.py``:

  * **Entry**: limit order priced like live aggressive limits
    (long: ref * (1 + 0.01%); short: ref * (1 - 0.01%)), same idea as
    ``ticker['ask'] * 1.0001`` / ``ticker['bid'] * 0.9999`` when OHLC
    is the only reference — we use the signal bar **close** as ref.
  * **Entry fill**: wait up to ``LIMIT_ORDER_TTL`` **trading-TF** bars;
    fill when price **touches** the limit (same idea as full_backtest).
    Adverse slippage on fill (``SLIPPAGE_BPS``).
  * **Entry fee**: **taker** (0.055%) — marketable / aggressive limits
    typically pay taker on Bybit.
  * **Exit**: TP at strategy limit level → **maker** (0.02%). SL / trail
    → **stop-market** → **taker** (0.055%). Timeout / forced close →
    **taker** (like ``close_position`` market order).
  * **SL / TP prices**: absolute levels from ``SignalEvent`` (not
    recomputed from fill) — matches attaching TP/SL to the order.
  * **Sizing denominator**: ``TAKER + TAKER + SLIPPAGE`` (conservative
    round-trip vs live risk_manager maker+taker+slip).
  * **Funding**: pro-rata by full 8h periods held (trading bar length
    from TF). Flat rate default; optional per-bar ``funding_8h`` from
    the **entry** bar when present in ``funding_series`` (indexed by
    entry bar index in collect).

This module does **not** simulate post-only / maker entry, partial fills,
liquidation, or exchange downtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.strategies.strategy_sniper import SniperStrategy
from src.strategies.base import _sf


def synthesize_volume_delta(df_5m: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate buy/sell split + volume_delta for backtest data that
    lacks tick-level trade flow (e.g. CCXT OHLCV cache).

    Uses the closing-position-within-bar as a proxy for buyer
    aggression (close near high → buy-dominant).  Also produces a
    volume_delta that drives ``order_flow_imb`` (OFI) into a
    realistic (≈0.30…0.70) range.

    Idempotent: if buy/sell volume already differs from a 50/50 split
    the frame is returned unchanged.
    """
    if df_5m is None or df_5m.empty:
        return df_5m
    needed = {"open", "high", "low", "close", "volume"}
    if not needed.issubset(df_5m.columns):
        return df_5m

    have_real_split = False
    if "buy_volume" in df_5m.columns and "sell_volume" in df_5m.columns:
        bv = df_5m["buy_volume"].astype(float)
        sv = df_5m["sell_volume"].astype(float)
        # If the cache already wrote anything other than the 50/50 default,
        # respect it.
        if (bv - sv).abs().sum() > 1e-9:
            have_real_split = True

    if have_real_split:
        return df_5m

    df = df_5m.copy()
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    loc = ((df["close"] - df["low"]) / rng).fillna(0.5).clip(0.0, 1.0)
    # Map close-in-range [0,1] → buy fraction [0.30, 0.70].  Open vs
    # close direction tilts another ±0.05 so reversals on the same
    # bar location still differ.
    body = (df["close"] - df["open"]).fillna(0.0)
    body_sign = body.where(body == 0, body / body.abs()).fillna(0.0)
    buy_frac = (0.30 + 0.40 * loc + 0.05 * body_sign).clip(0.05, 0.95)

    df["buy_volume"] = df["volume"] * buy_frac
    df["sell_volume"] = df["volume"] * (1.0 - buy_frac)
    df["volume_delta"] = df["buy_volume"] - df["sell_volume"]
    return df

# Bybit linear (typical) — see risk_manager / full_backtest
MAKER_FEE = 0.0002
TAKER_FEE = 0.00055
SLIPPAGE_BPS = 0.0003
# Live aggressive limit: order_manager uses ~0.01% through bid/ask
ENTRY_AGGRESSION_BPS = 0.0001

# Sizing: conservative vs maker entry assumption in risk_manager
ROUND_TRIP_COST_SIZING = TAKER_FEE + TAKER_FEE + SLIPPAGE_BPS

NOTIONAL_CAP_FRAC = 0.25
COOLDOWN_BARS = 6
LEVERAGE = 20
TRAIL_ACTIVATE = 0.10
TRAIL_OFFSET = 0.03
LIMIT_ORDER_TTL = 3
# Flat 8h funding accrual when per-bar funding missing (matches full_backtest)
FUNDING_RATE_8H_FLAT = 0.0001


@dataclass
class LivePortfolioTrade:
    """One resolved trade for portfolio replay."""

    symbol: str
    tf: str
    direction: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float  # fill price
    exit_price: float
    sl_distance: float  # abs(entry_fill - initial_sl) for logging
    tp_distance: float
    raw_pct: float  # price-only return from entry fill to exit
    exit_reason: str
    bars_held: int
    # Funding: if None, run_portfolio uses flat rate + bars_held + tf
    funding_rate_8h: float | None = None
    # ---- Partial-TP support ----
    # Portfolio runner: total_pnl = notional * (raw_pct * remaining_fraction
    #                                           + partial_pnl_pct)
    # Funding only accrues on the remaining notional after each rung fills,
    # but for tractability we approximate with average remaining notional
    # weighted by bar count between rungs.
    partial_pnl_pct: float = 0.0
    remaining_fraction: float = 1.0
    # Average remaining notional fraction across the lifetime of the trade,
    # used to scale funding cost.  1.0 if no partials hit.
    avg_remaining_fraction: float = 1.0
    # Fill model used at entry: "post_only"/"limit" → MAKER fee,
    # "market" → TAKER fee.  Defaults to TAKER for back-compat with
    # older trade producers that don't set this.
    entry_fill_mode: str = "market"


def _bar_minutes(tf: str) -> int:
    return 15 if tf == "15m" else 240


def _count_funding_fixes(
    entry_ts: pd.Timestamp, exit_ts: pd.Timestamp,
) -> int:
    """
    Count 8h funding fixes (00:00 / 08:00 / 16:00 UTC) in
    ``(entry_ts, exit_ts]``.  Mirrors live Bybit accrual.
    """
    if exit_ts <= entry_ts:
        return 0
    eight = pd.Timedelta(hours=8)
    next_fix = entry_ts.ceil("8h")
    if next_fix == entry_ts:
        next_fix = next_fix + eight
    if next_fix > exit_ts:
        return 0
    return int((exit_ts - next_fix) // eight) + 1


def _funding_usd(
    notional: float,
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    rate_8h: float | None,
    avg_remaining_fraction: float = 1.0,
) -> float:
    """
    Funding cost = (avg notional after partials) * rate_8h * #fixes_held.

    ``rate_8h`` is the rate observed at entry; positive means longs pay,
    shorts receive.  Sign convention matches live order_manager.
    """
    n_fix = _count_funding_fixes(entry_ts, exit_ts)
    if n_fix <= 0:
        return 0.0
    r = rate_8h if rate_8h is not None else FUNDING_RATE_8H_FLAT
    return notional * avg_remaining_fraction * r * n_fix


def _apply_exit_slippage(
    exit_price: float, direction: str, exit_reason: str,
) -> float:
    """Adverse slippage on stop-like exits; TP stays at limit."""
    if exit_reason == "tp":
        return exit_price
    if direction == "long":
        return exit_price * (1.0 - SLIPPAGE_BPS)
    return exit_price * (1.0 + SLIPPAGE_BPS)


def collect_trades_live_aligned(
    trading_bars: pd.DataFrame,
    context_df: pd.DataFrame | None,
    symbol: str,
    tf: str,
    snap_fn: Callable[[pd.Series, pd.Series], dict] | None = None,
    strategy: Any = None,
    chop_filter: Callable[[pd.Series, str], bool] | None = None,
) -> list[tuple[LivePortfolioTrade, dict]]:
    """
    Generate signals, simulate limit entry (TTL), then SL/TP/trail with
    partial-TP ladder and per-strategy cooldown.

    Parameters
    ----------
    trading_bars : DataFrame
        Trading-TF bars with indicator columns already attached.
    context_df : DataFrame | None
        Higher-TF context bars (e.g. 4h for a 15m trading TF).
    symbol, tf : str
        Logging metadata.
    snap_fn : callable, optional
        Snapshot extractor for filter sims.  Defaults to empty dict.
    strategy : BaseStrategy, optional
        Any concrete ``BaseStrategy`` instance.  If ``None``, defaults
        to ``SniperStrategy()`` for backwards compatibility with the
        existing portfolio sim.
    chop_filter : callable, optional
        ``(bar, direction) -> bool``.  Signals where this returns False
        are dropped before queueing.  Use it for the live anti-chop
        filter when the simulation should mirror live behaviour.
    """
    if strategy is None:
        strategy = SniperStrategy()
    out: list[tuple[LivePortfolioTrade, dict]] = []
    snap_fn = snap_fn or (lambda b, c: {})

    # Per-strategy cooldown / TP-ladder pulled from the strategy class.
    cooldown = getattr(strategy, "cooldown_bars", None) or COOLDOWN_BARS
    base_tp_ladder = getattr(strategy, "default_tp_ladder", None) or []
    move_be_on_tp1 = getattr(strategy, "move_be_on_tp1", True)

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
    n = len(trading_bars)

    funding_col = None
    if "funding_8h" in trading_bars.columns:
        funding_col = trading_bars["funding_8h"].values

    last_exit_idx = -cooldown - 1
    pending: dict | None = None
    pending_bars = 0

    for i in range(n):
        bar = trading_bars.iloc[i]
        ctx = get_ctx(pd.Timestamp(bar.get("timestamp")))

        if pending is not None:
            pending_bars += 1
            sig = pending["sig"]
            limit_px = pending["limit_px"]
            snap = pending["snap"]
            fr = pending["funding_rate_8h"]
            bh, bl = highs[i], lows[i]
            filled = False
            fill_price = 0.0

            # Resting maker fills hit at the resting price (we provide
            # liquidity); aggressive market fills cross the spread and
            # eat slippage.  Matches live OrderManager fee/slippage
            # accounting (post-only → maker, fallback market → taker).
            entry_slip = 0.0 if sig.effective_fill_mode() in (
                "post_only", "limit"
            ) else SLIPPAGE_BPS

            if sig.direction == "long":
                if bl <= limit_px:
                    fill_price = limit_px * (1.0 + entry_slip)
                    filled = True
            else:
                if bh >= limit_px:
                    fill_price = limit_px * (1.0 - entry_slip)
                    filled = True

            if filled:
                k = i
                entry_fill = fill_price
                sl_abs = float(sig.stop_loss)
                tp_abs = float(sig.take_profit)
                # Build per-trade ladder from strategy preference (signal
                # may carry its own override).
                tp_ladder = sig.tp_ladder or base_tp_ladder
                risk_dist = abs(entry_fill - sl_abs)
                sign = 1 if sig.direction == "long" else -1
                ladder_remaining: list[tuple[float, float]] = []
                if tp_ladder and risk_dist > 0:
                    for r_mult, frac in tp_ladder:
                        level = entry_fill + sign * r_mult * risk_dist
                        ladder_remaining.append((level, frac))
                    ladder_remaining.sort(
                        key=lambda t: t[0],
                        reverse=(sig.direction == "short"),
                    )

                trail_active = False
                hwm = 0.0
                current_sl = sl_abs
                be_activated = False
                remaining_fraction = 1.0
                partial_pnl_pct = 0.0  # weighted by *original* notional
                # Average remaining fraction (bar-weighted) for funding.
                rem_bar_accum = 0.0
                rem_bar_count = 0
                exit_price = None
                exit_reason = None
                exit_idx = n - 1

                for j in range(k + 1, n):
                    bh, bl = highs[j], lows[j]

                    # 1. Process partial-TP rungs touched on this bar.
                    while ladder_remaining:
                        level, frac = ladder_remaining[0]
                        hit_partial = (
                            bh >= level if sig.direction == "long"
                            else bl <= level
                        )
                        if not hit_partial:
                            break
                        if sig.direction == "long":
                            partial_raw = (level - entry_fill) / entry_fill
                        else:
                            partial_raw = (entry_fill - level) / entry_fill
                        partial_pnl_pct += partial_raw * frac
                        remaining_fraction = max(
                            0.0, remaining_fraction - frac,
                        )
                        if (move_be_on_tp1 and not be_activated
                                and current_sl != entry_fill):
                            current_sl = entry_fill
                            be_activated = True
                        ladder_remaining.pop(0)

                    if remaining_fraction <= 1e-6:
                        # Whole position closed by partials.  Reconstruct an
                        # average exit price for reporting.
                        if sig.direction == "long":
                            exit_price = entry_fill * (
                                1.0 + partial_pnl_pct
                            )
                        else:
                            exit_price = entry_fill * (
                                1.0 - partial_pnl_pct
                            )
                        exit_reason = "tp"
                        exit_idx = j
                        break

                    rem_bar_accum += remaining_fraction
                    rem_bar_count += 1

                    if sig.direction == "long":
                        gain = (bh - entry_fill) / entry_fill
                        if gain >= TRAIL_ACTIVATE:
                            trail_active = True
                        if trail_active:
                            if bh > hwm:
                                hwm = bh
                            new_sl = hwm * (1.0 - TRAIL_OFFSET)
                            if new_sl > current_sl:
                                current_sl = new_sl
                        sl_hit = bl <= current_sl
                        tp_hit = bh >= tp_abs
                    else:
                        gain = (entry_fill - bl) / entry_fill
                        if gain >= TRAIL_ACTIVATE:
                            trail_active = True
                        if trail_active:
                            if hwm == 0 or bl < hwm:
                                hwm = bl
                            new_sl = hwm * (1.0 + TRAIL_OFFSET)
                            if new_sl < current_sl:
                                current_sl = new_sl
                        sl_hit = bh >= current_sl
                        tp_hit = bl <= tp_abs

                    if sl_hit:
                        exit_price = _apply_exit_slippage(
                            current_sl, sig.direction,
                            ("trail" if trail_active
                             else "be" if be_activated else "sl"),
                        )
                        exit_reason = (
                            "trail" if trail_active
                            else "be" if be_activated else "sl"
                        )
                        exit_idx = j
                        break
                    if tp_hit:
                        exit_price = _apply_exit_slippage(
                            tp_abs, sig.direction, "tp",
                        )
                        exit_reason = "tp"
                        exit_idx = j
                        break

                if exit_price is None:
                    exit_price = _apply_exit_slippage(
                        closes[-1], sig.direction, "timeout",
                    )
                    exit_reason = "timeout"
                    exit_idx = n - 1

                if sig.direction == "long":
                    raw_pct = (exit_price - entry_fill) / entry_fill
                else:
                    raw_pct = (entry_fill - exit_price) / entry_fill

                avg_remaining = (
                    rem_bar_accum / rem_bar_count
                    if rem_bar_count > 0 else remaining_fraction
                )

                last_exit_idx = exit_idx
                pending = None
                pending_bars = 0

                sl_dist = abs(entry_fill - sl_abs)
                tp_dist = abs(tp_abs - entry_fill)

                trade = LivePortfolioTrade(
                    symbol=symbol,
                    tf=tf,
                    direction=sig.direction,
                    entry_time=pd.Timestamp(timestamps[k]),
                    exit_time=pd.Timestamp(timestamps[exit_idx]),
                    entry_price=entry_fill,
                    exit_price=exit_price,
                    sl_distance=sl_dist,
                    tp_distance=tp_dist,
                    raw_pct=raw_pct,
                    exit_reason=exit_reason,
                    bars_held=exit_idx - k,
                    funding_rate_8h=fr,
                    partial_pnl_pct=partial_pnl_pct,
                    remaining_fraction=remaining_fraction,
                    avg_remaining_fraction=avg_remaining,
                    entry_fill_mode=sig.effective_fill_mode() or "post_only",
                )
                out.append((trade, snap))
                continue

            if pending_bars >= LIMIT_ORDER_TTL:
                pending = None
                pending_bars = 0
            continue

        if i <= last_exit_idx + cooldown:
            continue

        sig = strategy.generate_signal(
            symbol=symbol,
            indicators_5m=bar,
            indicators_15m=ctx,
            funding_rate=0.0,
            liq_volume_1h=0.0,
        )
        if sig is None or sig.direction == "flat":
            continue

        if chop_filter is not None and not chop_filter(bar, sig.direction):
            continue

        close = _sf(bar.get("close"))
        if close <= 0:
            continue

        # Respect strategy fill_mode preference: post_only / limit ⇒ rest at
        # the signal price (passive); market ⇒ pay aggressive limit through
        # the spread (matches live taker fallback in OrderManager).
        fill_mode = sig.effective_fill_mode() or "post_only"
        if fill_mode == "market":
            if sig.direction == "long":
                limit_px = close * (1.0 + ENTRY_AGGRESSION_BPS)
            else:
                limit_px = close * (1.0 - ENTRY_AGGRESSION_BPS)
        else:
            limit_px = float(sig.entry_price) if sig.entry_price > 0 else close

        fr = None
        if funding_col is not None and i < len(funding_col):
            v = funding_col[i]
            if v == v and float(v) != 0.0:
                fr = float(v)

        snap = snap_fn(bar, ctx)
        pending = {
            "sig": sig,
            "limit_px": limit_px,
            "snap": snap,
            "funding_rate_8h": fr,
        }
        pending_bars = 0

    return out


def run_portfolio_live_aligned(
    trades: list[LivePortfolioTrade],
    *,
    start_equity: float,
    risk_pct: float,
    max_concurrent: int = 4,
    track_equity_curve: bool = False,
    curve_from: str | None = None,
) -> dict:
    """Shared equity; fees and funding match live assumptions."""
    equity = start_equity
    peak = equity
    max_dd_pct = 0.0
    max_dd_usd = 0.0
    min_equity = equity
    equity_curve: list = []
    if track_equity_curve and curve_from:
        equity_curve.append((pd.Timestamp(curve_from), equity))

    events = []
    for idx, t in enumerate(trades):
        events.append(("entry", t.entry_time, idx))
        events.append(("exit", t.exit_time, idx))
    events.sort(key=lambda e: (e[1], 0 if e[0] == "exit" else 1))

    open_positions: dict[int, dict] = {}
    closed = wins = 0
    total_pnl = 0.0
    total_fees = 0.0
    total_funding = 0.0
    open_symbols: set[str] = set()
    skipped = 0
    max_concurrent_seen = 0
    tp_count = sl_count = trail_count = timeout_count = 0

    def exit_fee_rate(reason: str) -> float:
        return MAKER_FEE if reason == "tp" else TAKER_FEE

    for event_type, event_time, trade_idx in events:
        trade = trades[trade_idx]

        if event_type == "exit" and trade_idx in open_positions:
            pos = open_positions.pop(trade_idx)
            notional = pos["notional"]
            entry_fee = pos["entry_fee"]
            efr = exit_fee_rate(trade.exit_reason)
            # Partial-TP rungs are limit fills -> maker fee on the
            # closed fraction; the remainder pays the exit-reason fee.
            partial_frac = max(0.0, 1.0 - trade.remaining_fraction)
            partial_fee = notional * partial_frac * MAKER_FEE
            exit_fee = notional * trade.remaining_fraction * efr
            fund = _funding_usd(
                notional, trade.entry_time, trade.exit_time,
                trade.funding_rate_8h,
                avg_remaining_fraction=trade.avg_remaining_fraction,
            )
            pnl = (
                notional * (
                    trade.raw_pct * trade.remaining_fraction
                    + trade.partial_pnl_pct
                )
                - entry_fee
                - partial_fee
                - exit_fee
                - fund
            )
            equity += pnl
            total_pnl += pnl
            total_fees += entry_fee + exit_fee
            total_funding += fund
            total_fees += partial_fee
            closed += 1
            if pnl > 0:
                wins += 1
            if trade.exit_reason == "tp":
                tp_count += 1
            elif trade.exit_reason == "trail":
                trail_count += 1
            elif trade.exit_reason in ("sl", "be"):
                sl_count += 1
            elif trade.exit_reason == "timeout":
                timeout_count += 1
            open_symbols.discard(f"{trade.symbol}_{trade.tf}")
            if equity > peak:
                peak = equity
            dd_pct = (peak - equity) / peak if peak > 0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_usd = peak - equity
            if equity < min_equity:
                min_equity = equity
            if track_equity_curve:
                equity_curve.append((event_time, equity))

        elif event_type == "entry":
            sym_key = f"{trade.symbol}_{trade.tf}"
            if (
                len(open_positions) >= max_concurrent
                or sym_key in open_symbols
                or equity <= 0
            ):
                skipped += 1
                continue

            entry = trade.entry_price
            sl_dist = trade.sl_distance
            if sl_dist <= 0:
                skipped += 1
                continue

            risk_amount = equity * risk_pct
            qty = risk_amount / (sl_dist + entry * ROUND_TRIP_COST_SIZING)
            notional = qty * entry
            max_notional = equity * LEVERAGE * NOTIONAL_CAP_FRAC
            if notional > max_notional:
                notional = max_notional

            # Maker fee for resting limit / post-only fills, taker for
            # aggressive market entries.  Matches live OrderManager which
            # tries post-only first and falls back to market on TTL expiry.
            entry_fee_rate = (
                MAKER_FEE if trade.entry_fill_mode in ("post_only", "limit")
                else TAKER_FEE
            )
            entry_fee = notional * entry_fee_rate
            open_positions[trade_idx] = {
                "notional": notional,
                "entry_fee": entry_fee,
            }
            open_symbols.add(sym_key)
            if len(open_positions) > max_concurrent_seen:
                max_concurrent_seen = len(open_positions)

    for trade_idx, pos in list(open_positions.items()):
        trade = trades[trade_idx]
        notional = pos["notional"]
        efr = exit_fee_rate(trade.exit_reason)
        partial_frac = max(0.0, 1.0 - trade.remaining_fraction)
        partial_fee = notional * partial_frac * MAKER_FEE
        exit_fee = notional * trade.remaining_fraction * efr
        fund = _funding_usd(
            notional, trade.entry_time, trade.exit_time,
            trade.funding_rate_8h,
            avg_remaining_fraction=trade.avg_remaining_fraction,
        )
        pnl = (
            notional * (
                trade.raw_pct * trade.remaining_fraction
                + trade.partial_pnl_pct
            )
            - pos["entry_fee"]
            - partial_fee
            - exit_fee
            - fund
        )
        equity += pnl
        total_pnl += pnl
        total_fees += pos["entry_fee"] + partial_fee + exit_fee
        total_funding += fund
        closed += 1
        if pnl > 0:
            wins += 1
        if equity > peak:
            peak = equity
        dd_pct = (peak - equity) / peak if peak > 0 else 0.0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_usd = peak - equity
        if equity < min_equity:
            min_equity = equity
        if track_equity_curve:
            equity_curve.append((trade.exit_time, equity))

    if equity > peak:
        peak = equity
    dd_pct = (peak - equity) / peak if peak > 0 else 0.0
    if dd_pct > max_dd_pct:
        max_dd_pct = dd_pct
        max_dd_usd = peak - equity
    if equity < min_equity:
        min_equity = equity

    wr = wins / closed * 100 if closed else 0.0
    out = {
        "trades": closed,
        "wins": wins,
        "wr": wr,
        "pnl": total_pnl,
        "final_eq": equity,
        "total_fees": total_fees,
        "total_funding": total_funding,
        "max_dd_pct": max_dd_pct,
        "max_dd_usd": max_dd_usd,
        "min_equity": min_equity,
        "skipped": skipped,
        "max_concurrent": max_concurrent_seen,
        "tp": tp_count,
        "sl": sl_count,
        "trail": trail_count,
        "timeout": timeout_count,
    }
    if track_equity_curve:
        out["equity_curve"] = equity_curve
    return out
