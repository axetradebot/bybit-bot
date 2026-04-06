"""
SIM ONLY: maximize final portfolio equity subject to max drawdown < cap.

Reuses live-aligned collection from anti_chop_sweep_sim and combo filters from
pnl_loss_tradeoff_sim. Sweeps:
  - every baseline + combo scenario (post-hoc signal filter)
  - several risk-per-trade fractions (main lever for DD vs return)

Objective: best final $ among runs with max_dd_pct < --max-dd (default 65%),
or with --no-dd-cap: rank purely by final equity / end PnL (DD reported only).

Run:
    python scripts/pnl_dd_constrained_opt_sim.py
    python scripts/pnl_dd_constrained_opt_sim.py --no-dd-cap
    python scripts/pnl_dd_constrained_opt_sim.py --max-dd 0.50
    python scripts/pnl_dd_constrained_opt_sim.py --risk-grid 0.005,0.01,0.015,0.02
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

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


def _load_pnl_loss():
    p = Path(__file__).resolve().parent / "pnl_loss_tradeoff_sim.py"
    name = "pnl_loss_tradeoff_sim"
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ac = _load_anti_chop()
pl = _load_pnl_loss()

START_EQUITY = ac.START_EQUITY
SYMBOLS = ac.SYMBOLS
TIMEFRAMES = ac.TIMEFRAMES
CONTEXT_TF = ac.CONTEXT_TF
MAX_CONCURRENT = ac.MAX_CONCURRENT
build_bars_for_tf = ac.build_bars_for_tf
collect_trades_with_snapshots = ac.collect_trades_with_snapshots
load_candles = ac.load_candles
build_combo_scenarios = pl.build_combo_scenarios

from src.backtest.live_aligned_portfolio import run_portfolio_live_aligned


DEFAULT_RISK_GRID = (0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025)
# Extra risk levels when --no-dd-cap (search for max end PnL).
UNCAPPED_EXTRA_RISK = (0.03, 0.035, 0.04, 0.05)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Maximize final equity (optional max drawdown cap).",
    )
    p.add_argument(
        "--no-dd-cap",
        action="store_true",
        help="Do not filter by drawdown; maximize final equity only. "
        "Default risk grid adds 3%%–5%% steps unless --risk-grid is set.",
    )
    p.add_argument(
        "--max-dd",
        type=float,
        default=0.65,
        help="Maximum allowed peak-to-trough drawdown fraction (ignored with --no-dd-cap)",
    )
    p.add_argument(
        "--risk-grid",
        type=str,
        default="",
        help="Comma-separated risk fractions e.g. 0.005,0.01,0.015 "
        f"(default: {','.join(str(x) for x in DEFAULT_RISK_GRID)})",
    )
    p.add_argument(
        "--no-combos",
        action="store_true",
        help="Only baseline anti_chop scenarios (faster, smaller search)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cap: float | None
    if args.no_dd_cap:
        cap = None
    else:
        cap = args.max_dd
        if cap <= 0 or cap >= 1:
            print("--max-dd should be in (0, 1), e.g. 0.65 for 65%", flush=True)
            sys.exit(1)

    if args.risk_grid.strip():
        risk_grid = tuple(
            float(x.strip())
            for x in args.risk_grid.split(",")
            if x.strip()
        )
    elif args.no_dd_cap:
        risk_grid = DEFAULT_RISK_GRID + UNCAPPED_EXTRA_RISK
    else:
        risk_grid = DEFAULT_RISK_GRID

    base_scenarios = ac.build_scenarios()
    combo_scenarios = [] if args.no_combos else build_combo_scenarios()
    scenarios = base_scenarios + combo_scenarios

    cache_dir = project_root / "data_cache"

    print("=" * 110, flush=True)
    if cap is None:
        print("MAX END PnL / FINAL EQUITY - no DD cap (live-aligned sim)", flush=True)
        print(
            f"Start equity ${START_EQUITY:,.0f}  |  max concurrent {MAX_CONCURRENT}  |  "
            f"DD cap: none  |  risk grid {risk_grid}",
            flush=True,
        )
    else:
        print("PnL OPTIMIZATION UNDER MAX DRAWDOWN (live-aligned sim)", flush=True)
        print(
            f"Start equity ${START_EQUITY:,.0f}  |  max concurrent {MAX_CONCURRENT}  |  "
            f"DD cap {cap:.1%}  |  risk grid {risk_grid}",
            flush=True,
        )
    print(f"Scenarios: {len(base_scenarios)} base + {len(combo_scenarios)} combo = {len(scenarios)}", flush=True)
    print("=" * 110, flush=True)

    print("\n-- Collect baseline trades + snapshots --", flush=True)
    rows: list[tuple[ac.PrecomputedTrade, dict]] = []
    for sym in SYMBOLS:
        df_5m = load_candles(sym, cache_dir)
        if df_5m.empty:
            print(f"  skip {sym} (no cache)", flush=True)
            continue
        for tf in TIMEFRAMES:
            t0 = time.time()
            ctx_df = build_bars_for_tf(df_5m, CONTEXT_TF.get(tf, "4h"))
            bars = build_bars_for_tf(df_5m, tf)
            chunk = collect_trades_with_snapshots(bars, ctx_df, sym, tf)
            rows.extend(chunk)
            print(f"  {tf} {sym}: {len(chunk)} ({time.time()-t0:.1f}s)", flush=True)

    rows.sort(key=lambda x: x[0].entry_time)
    print(f"\n  Total baseline trades: {len(rows)}", flush=True)

    print("\n-- Grid search: scenario x risk_pct --", flush=True)
    results: list[dict] = []
    t_grid = time.time()
    for label, pred in scenarios:
        filt = [t for t, snap in rows if pred(snap, t.direction)]
        for rp in risk_grid:
            r = run_portfolio_live_aligned(
                filt,
                start_equity=START_EQUITY,
                risk_pct=rp,
                max_concurrent=MAX_CONCURRENT,
            )
            results.append({
                "label": label,
                "risk_pct": rp,
                "kept": len(filt),
                "dd": r["max_dd_pct"],
                "final_eq": r["final_eq"],
                "pnl": r["pnl"],
                "trades": r["trades"],
                "wr": r["wr"],
                "feasible": cap is None or r["max_dd_pct"] < cap,
            })

    print(f"  Evaluated {len(results)} configs in {time.time()-t_grid:.1f}s", flush=True)

    feasible = [x for x in results if x["feasible"]]
    infeasible = [x for x in results if not x["feasible"]]

    print("\n" + "=" * 110, flush=True)
    if feasible:
        feasible.sort(key=lambda x: x["final_eq"], reverse=True)
        best = feasible[0]
        if cap is None:
            print(
                f"BEST (max final equity):  {best['label']!r}  |  risk {best['risk_pct']:.2%}",
                flush=True,
            )
        else:
            print(
                f"BEST UNDER DD < {cap:.1%}:  {best['label']!r}  |  risk {best['risk_pct']:.2%}",
                flush=True,
            )
        print(
            f"  final ${best['final_eq']:,.0f}  |  PnL ${best['pnl']:+,.0f}  |  "
            f"max DD {best['dd']:.2%}  |  closed {best['trades']}  |  kept signals {best['kept']}",
            flush=True,
        )
        hdr = (
            f"{'Rank':>4}  {'risk':>7}  {'Final$':>10}  {'PnL':>11}  {'MaxDD':>8}  "
            f"{'Tr':>6}  {'Scenario':<50}"
        )
        top_title = "TOP 40 BY FINAL EQUITY" if cap is None else "TOP 40 FEASIBLE (by final equity)"
        print(f"\n{top_title}", flush=True)
        print(hdr, flush=True)
        print("-" * 110, flush=True)
        for i, x in enumerate(feasible[:40], 1):
            print(
                f"{i:>4}  {x['risk_pct']:>6.2%}  ${x['final_eq']:>8,.0f}  "
                f"${x['pnl']:>+9,.0f}  {x['dd']:>7.1%}  {x['trades']:>6d}  "
                f"{x['label']:<50}",
                flush=True,
            )
    else:
        print(f"NO CONFIGURATION satisfies max DD < {cap:.1%}", flush=True)
        if infeasible:
            soft = min(infeasible, key=lambda x: x["dd"])
            hi = max(infeasible, key=lambda x: x["final_eq"])
            print(
                f"  Tightest DD: {soft['dd']:.1%} ({soft['label']!r} @ {soft['risk_pct']:.2%})",
                flush=True,
            )
            print(
                f"  Highest PnL (violates cap): ${hi['final_eq']:,.0f}  "
                f"DD {hi['dd']:.1%} ({hi['label']!r} @ {hi['risk_pct']:.2%})",
                flush=True,
            )
            print(
                "  Try: raise --max-dd, lower risk grid max, or add stricter filters.",
                flush=True,
            )

    # Just above cap: best equity among DD in [cap, cap+5pp) (informational)
    near = (
        [x for x in results if cap <= x["dd"] < cap + 0.05]
        if cap is not None
        else []
    )
    if near:
        near.sort(key=lambda x: x["final_eq"], reverse=True)
        print("\n-- Slightly over DD cap [cap, cap+5pp): top 5 by final equity --", flush=True)
        for x in near[:5]:
            print(
                f"  DD {x['dd']:.1%}  risk {x['risk_pct']:.2%}  "
                f"${x['final_eq']:,.0f}  {x['label'][:60]}",
                flush=True,
            )

    print("=" * 110, flush=True)


if __name__ == "__main__":
    main()
