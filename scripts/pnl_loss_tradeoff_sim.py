"""
SIM ONLY: Max PnL vs cutting losses without (ideally) cutting winners.

For each filter we report:
  - Winner retention: share of baseline *price-win* trades (raw_pct>0) still taken
  - Loser cut rate: share of baseline losing trades filtered out
  - Portfolio final equity (shared $3k, same engine as anti_chop_sweep_sim)

Adds many combo scenarios on top of anti_chop_sweep_sim.build_scenarios().

Run:
    python scripts/pnl_loss_tradeoff_sim.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Callable

os.environ["PYTHONUNBUFFERED"] = "1"

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))


def _load_anti_chop():
    p = Path(__file__).resolve().parent / "anti_chop_sweep_sim.py"
    name = "anti_chop_sweep_sim"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ac = _load_anti_chop()

START_EQUITY = ac.START_EQUITY
SYMBOLS = ac.SYMBOLS
TIMEFRAMES = ac.TIMEFRAMES
CONTEXT_TF = ac.CONTEXT_TF
build_bars_for_tf = ac.build_bars_for_tf
collect_trades_with_snapshots = ac.collect_trades_with_snapshots
run_portfolio = ac.run_portfolio
load_candles = ac.load_candles
di_aligned = ac.di_aligned
macd_aligned = ac.macd_aligned


def bb_mid_side(s: dict, d: str) -> bool:
    return s["bb_mid"] > 0 and (
        (d == "long" and s["close"] > s["bb_mid"]) or
        (d == "short" and s["close"] < s["bb_mid"]))


def mfi_bias(s: dict, d: str) -> bool:
    return (d == "long" and s["mfi_14"] > 42) or (d == "short" and s["mfi_14"] < 58)


def bar_range_atr(s: dict, d: str) -> bool:
    return s["atr_14"] > 0 and (s["high"] - s["low"]) >= 0.45 * s["atr_14"]


def vwap_side(s: dict, d: str) -> bool:
    return s["vwap"] > 0 and (
        (d == "long" and s["close"] > s["vwap"]) or
        (d == "short" and s["close"] < s["vwap"]))


def vol_floor(s: dict, d: str, x: float) -> bool:
    v = s["volume_ratio"]
    return v >= x if v > 0 else True


def no_squeeze(s: dict, d: str) -> bool:
    return not s["bb_squeeze"]


def build_combo_scenarios() -> list[tuple[str, Callable[[dict, str], bool]]]:
    """High-PnL-oriented combinations from prior sweep winners."""
    c: list[tuple[str, Callable[[dict, str], bool]]] = []

    def add(name: str, pred: Callable[[dict, str], bool]) -> None:
        c.append((name, pred))

    # Pairs
    add("COMBO: DI + BB mid", lambda s, d: di_aligned(s, d) and bb_mid_side(s, d))
    add("COMBO: DI + MFI bias", lambda s, d: di_aligned(s, d) and mfi_bias(s, d))
    add("COMBO: DI + bar range 0.45 ATR", lambda s, d: di_aligned(s, d) and bar_range_atr(s, d))
    add("COMBO: DI + VWAP side", lambda s, d: di_aligned(s, d) and vwap_side(s, d))
    add("COMBO: DI + skip squeeze", lambda s, d: di_aligned(s, d) and no_squeeze(s, d))
    add("COMBO: DI + vol>=1.0", lambda s, d: di_aligned(s, d) and vol_floor(s, d, 1.0))
    add("COMBO: DI + vol>=1.2", lambda s, d: di_aligned(s, d) and vol_floor(s, d, 1.2))
    add("COMBO: BB mid + MFI", lambda s, d: bb_mid_side(s, d) and mfi_bias(s, d))
    add("COMBO: BB mid + bar range", lambda s, d: bb_mid_side(s, d) and bar_range_atr(s, d))
    add("COMBO: BB mid + VWAP", lambda s, d: bb_mid_side(s, d) and vwap_side(s, d))
    add("COMBO: MFI + bar range", lambda s, d: mfi_bias(s, d) and bar_range_atr(s, d))
    add("COMBO: DI + MACD 5m+15m", lambda s, d: di_aligned(s, d) and (
        macd_aligned(s["macd_hist"], d) and s["macd_hist_ctx"] != 0
        and macd_aligned(s["macd_hist_ctx"], d)))

    # Triples
    add("COMBO: DI + BB mid + MFI", lambda s, d: (
        di_aligned(s, d) and bb_mid_side(s, d) and mfi_bias(s, d)))
    add("COMBO: DI + BB mid + bar range", lambda s, d: (
        di_aligned(s, d) and bb_mid_side(s, d) and bar_range_atr(s, d)))
    add("COMBO: DI + BB mid + VWAP", lambda s, d: (
        di_aligned(s, d) and bb_mid_side(s, d) and vwap_side(s, d)))
    add("COMBO: DI + MFI + bar range", lambda s, d: (
        di_aligned(s, d) and mfi_bias(s, d) and bar_range_atr(s, d)))
    add("COMBO: DI + BB mid + vol>=1.0", lambda s, d: (
        di_aligned(s, d) and bb_mid_side(s, d) and vol_floor(s, d, 1.0)))
    add("COMBO: DI + BB mid + no squeeze", lambda s, d: (
        di_aligned(s, d) and bb_mid_side(s, d) and no_squeeze(s, d)))
    add("COMBO: BB mid + MFI + bar range", lambda s, d: (
        bb_mid_side(s, d) and mfi_bias(s, d) and bar_range_atr(s, d)))

    # Quad
    add("COMBO: DI + BB mid + MFI + bar range", lambda s, d: (
        di_aligned(s, d) and bb_mid_side(s, d) and mfi_bias(s, d) and bar_range_atr(s, d)))
    add("COMBO: DI + BB mid + MFI + vol>=1.0", lambda s, d: (
        di_aligned(s, d) and bb_mid_side(s, d) and mfi_bias(s, d) and vol_floor(s, d, 1.0)))

    # DI + atr rank (trend vol) + BB mid
    add("COMBO: DI + BB mid + atr_rank>=0.35", lambda s, d: (
        di_aligned(s, d) and bb_mid_side(s, d) and s["atr_pct_rank"] >= 0.35))

    # Softer BB: bb_width + DI + MFI
    add("COMBO: DI + MFI + bb_width/close>1.2%", lambda s, d: (
        di_aligned(s, d) and mfi_bias(s, d) and s["close"] > 0
        and s["bb_width"] / s["close"] > 0.012))

    return c


def signal_level_stats(
    rows: list[tuple[ac.PrecomputedTrade, dict]],
    pred: Callable[[dict, str], bool],
) -> dict:
    """How filter treats resolved baseline trades (raw PnL before portfolio path)."""
    nw = nl = 0
    kw = kl = 0  # kept winners, kept losers
    for t, snap in rows:
        win = t.raw_pct > 0
        if win:
            nw += 1
        else:
            nl += 1
        if not pred(snap, t.direction):
            continue
        if win:
            kw += 1
        else:
            kl += 1
    dw, dl = nw - kw, nl - kl
    winner_ret = kw / nw if nw else 0.0
    loser_cut = dl / nl if nl else 0.0
    # Among dropped trades, what fraction were losers? (precision of "removal")
    dropped = dw + dl
    precision = dl / dropped if dropped else 0.0
    return {
        "nw": nw, "nl": nl, "kw": kw, "kl": kl, "dw": dw, "dl": dl,
        "winner_ret": winner_ret, "loser_cut": loser_cut,
        "drop_precision": precision,
        "efficiency": winner_ret * loser_cut,
    }


def main():
    cache_dir = project_root / "data_cache"
    base_scenarios = ac.build_scenarios()
    combo_scenarios = build_combo_scenarios()
    scenarios = base_scenarios + combo_scenarios

    print("=" * 120, flush=True)
    print("PnL vs LOSS-CUT TRADEOFF (sim only)", flush=True)
    print(f"Baseline scenarios: {len(base_scenarios)}  |  Extra combos: {len(combo_scenarios)}  "
          f"|  Total: {len(scenarios)}", flush=True)
    print("=" * 120, flush=True)

    print("\n-- Collect baseline trades --", flush=True)
    rows: list[tuple[ac.PrecomputedTrade, dict]] = []
    for sym in SYMBOLS:
        df_5m = load_candles(sym, cache_dir)
        if df_5m.empty:
            print(f"  skip {sym}", flush=True)
            continue
        for tf in TIMEFRAMES:
            t0 = time.time()
            ctx_df = build_bars_for_tf(df_5m, CONTEXT_TF.get(tf, "4h"))
            bars = build_bars_for_tf(df_5m, tf)
            chunk = collect_trades_with_snapshots(bars, ctx_df, sym, tf)
            rows.extend(chunk)
            print(f"  {tf} {sym}: {len(chunk)} ({time.time()-t0:.1f}s)", flush=True)

    rows.sort(key=lambda x: x[0].entry_time)
    print(f"\n  Total trades: {len(rows)}", flush=True)

    print("\n-- Running scenarios --", flush=True)
    results = []
    for label, pred in scenarios:
        st = signal_level_stats(rows, pred)
        filt = [t for t, snap in rows if pred(snap, t.direction)]
        r = run_portfolio(filt)
        results.append({
            "label": label, "kept": len(filt), "st": st, "port": r,
        })

    # Table 1: max PnL
    by_pnl = sorted(results, key=lambda x: x["port"]["final_eq"], reverse=True)
    hdr = (
        f"{'Rank':>4}  {'Scenario':<52s}  {'Final$':>10s}  {'WinRet':>7s}  "
        f"{'LoseCut':>8s}  {'DropPrec':>8s}  {'Eff':>6s}  {'DD':>6s}  "
        f"{'dW':>5s}  {'dL':>5s}"
    )
    print("\n" + "=" * 120, flush=True)
    print("TOP 25 BY PORTFOLIO FINAL EQUITY (max PnL in this sim)", flush=True)
    print(hdr, flush=True)
    print("-" * 120, flush=True)
    for i, x in enumerate(by_pnl[:25], 1):
        st = x["st"]
        p = x["port"]
        print(
            f"{i:>4}  {x['label']:<52s}  ${p['final_eq']:>8,.0f}  "
            f"{st['winner_ret']:>6.1%}  {st['loser_cut']:>7.1%}  "
            f"{st['drop_precision']:>7.1%}  {st['efficiency']:>5.2f}  "
            f"{p['max_dd_pct']:>5.1%}  {st['dw']:>5d}  {st['dl']:>5d}",
            flush=True,
        )

    # Table 2: best loser cut while keeping >= 95% winners (signal level)
    constrained = [x for x in results if x["st"]["winner_ret"] >= 0.95]
    by_lcut = sorted(constrained, key=lambda x: (-x["st"]["loser_cut"], x["port"]["final_eq"]))
    print("\n" + "=" * 120, flush=True)
    print("SCENARIOS WITH >=95% WINNER RETENTION (signal level) - ranked by loser cut, then PnL", flush=True)
    print(hdr, flush=True)
    print("-" * 120, flush=True)
    if not constrained:
        print("  (none)", flush=True)
    else:
        for i, x in enumerate(by_lcut[:20], 1):
            st = x["st"]
            p = x["port"]
            print(
                f"{i:>4}  {x['label']:<52s}  ${p['final_eq']:>8,.0f}  "
                f"{st['winner_ret']:>6.1%}  {st['loser_cut']:>7.1%}  "
                f"{st['drop_precision']:>7.1%}  {st['efficiency']:>5.2f}  "
                f"{p['max_dd_pct']:>5.1%}  {st['dw']:>5d}  {st['dl']:>5d}",
                flush=True,
            )

    # Table 3: best composite efficiency * sqrt(final_eq) or just efficiency then filter top PnL
    by_eff = sorted(
        [x for x in results if x["kept"] > 500],
        key=lambda x: (x["st"]["efficiency"], x["port"]["final_eq"]),
        reverse=True,
    )
    print("\n" + "=" * 120, flush=True)
    print("TOP 15 BY SIGNAL EFFICIENCY (WinRet * LoserCut), min 500 kept signals", flush=True)
    print(hdr, flush=True)
    print("-" * 120, flush=True)
    for i, x in enumerate(by_eff[:15], 1):
        st = x["st"]
        p = x["port"]
        print(
            f"{i:>4}  {x['label']:<52s}  ${p['final_eq']:>8,.0f}  "
            f"{st['winner_ret']:>6.1%}  {st['loser_cut']:>7.1%}  "
            f"{st['drop_precision']:>7.1%}  {st['efficiency']:>5.2f}  "
            f"{p['max_dd_pct']:>5.1%}  {st['dw']:>5d}  {st['dl']:>5d}",
            flush=True,
        )

    print("\n" + "=" * 120, flush=True)
    print("Legend: WinRet = kept winners / all winners  |  LoseCut = dropped losers / all losers", flush=True)
    print("        DropPrec = dropped losers / all dropped  |  Eff = WinRet * LoseCut  |  dW/dL = dropped W/L counts")
    print("=" * 120, flush=True)


if __name__ == "__main__":
    main()
