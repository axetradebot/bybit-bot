"""Steps 3-4 of the re-run: recompute indicators, run backtests, compare."""
import sys, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import create_engine, text
from src.config import settings

engine = create_engine(settings.sync_db_url)

print("[3/4] Recomputing indicators (MACD signal=3 fix)...")
for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
    print(f"  -> {sym} ...", end=" ", flush=True)
    r = subprocess.run(
        [sys.executable, "src/indicators/compute_all.py", "--symbol", sym],
        capture_output=True, text=True, encoding="utf-8",
    )
    if r.returncode != 0:
        print(f"FAILED: {r.stderr[-300:]}")
        sys.exit(1)
    print("OK")

print("\n[4/4] Running backtests...")
strategies = [
    "bb_squeeze", "multitf_scalp", "rsi_divergence",
    "vwap_reversion", "volume_delta_liq", "regime_adaptive",
]
for strat in strategies:
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        print(f"  -> {strat} / {sym} ...", end=" ", flush=True)
        r = subprocess.run(
            [sys.executable, "src/backtest/simulator.py",
             "--strategy", strat, "--symbol", sym],
            capture_output=True, text=True, encoding="utf-8",
        )
        last = (r.stdout.strip().split("\n")[-1] if r.stdout.strip() else "no output")
        if r.returncode != 0:
            err = r.stderr.strip().split("\n")[-1] if r.stderr.strip() else ""
            print(f"FAIL: {err}")
        else:
            print(last)

print("\nUpdating strategy_performance...")
subprocess.run(
    [sys.executable, "scripts/update_strategy_performance.py"],
    capture_output=True, text=True, encoding="utf-8",
)

print("\n" + "=" * 80)
print("COMPARISON: Before vs After regime filters + MACD fix")
print("=" * 80)
with engine.connect() as conn:
    print("\n--- BEFORE (pre-regime) ---")
    old_rows = conn.execute(text("""
        SELECT strategy_combo, symbol,
               COUNT(*) AS trades,
               ROUND(AVG(win_loss::int) * 100, 1) AS wr,
               ROUND(AVG(pnl_pct) * 100, 4) AS avg_pnl,
               ROUND(SUM(pnl_usd), 2) AS total_pnl
        FROM trades_log_pre_regime
        GROUP BY strategy_combo, symbol
        HAVING COUNT(*) >= 10
        ORDER BY AVG(pnl_pct) DESC
    """)).fetchall()

    print(f"{'Combo':<55} {'Sym':<10} {'N':>6} {'WR%':>6} {'AvgPnl%':>8} {'TotalPnL':>11}")
    print("-" * 100)
    for r in old_rows:
        c = str(r[0])[:54]
        print(f"{c:<55} {r[1]:<10} {r[2]:>6} {r[3]:>5.1f}% {r[4]:>7.4f}% ${r[5]:>10.2f}")

    print("\n--- AFTER (with regime filters + MACD fix) ---")
    new_rows = conn.execute(text("""
        SELECT strategy_combo, symbol,
               COUNT(*) AS trades,
               ROUND(AVG(win_loss::int) * 100, 1) AS wr,
               ROUND(AVG(pnl_pct) * 100, 4) AS avg_pnl,
               ROUND(SUM(pnl_usd), 2) AS total_pnl
        FROM trades_log
        WHERE is_backtest = TRUE
        GROUP BY strategy_combo, symbol
        HAVING COUNT(*) >= 10
        ORDER BY AVG(pnl_pct) DESC
    """)).fetchall()

    print(f"{'Combo':<55} {'Sym':<10} {'N':>6} {'WR%':>6} {'AvgPnl%':>8} {'TotalPnL':>11}")
    print("-" * 100)
    for r in new_rows:
        c = str(r[0])[:54]
        print(f"{c:<55} {r[1]:<10} {r[2]:>6} {r[3]:>5.1f}% {r[4]:>7.4f}% ${r[5]:>10.2f}")

    print("\n--- AGGREGATE ---")
    old_a = conn.execute(text(
        "SELECT COUNT(*), ROUND(AVG(win_loss::int)*100,1), "
        "ROUND(AVG(pnl_pct)*100,4), ROUND(SUM(pnl_usd),2) "
        "FROM trades_log_pre_regime"
    )).fetchone()
    new_a = conn.execute(text(
        "SELECT COUNT(*), ROUND(AVG(win_loss::int)*100,1), "
        "ROUND(AVG(pnl_pct)*100,4), ROUND(SUM(pnl_usd),2) "
        "FROM trades_log WHERE is_backtest = TRUE"
    )).fetchone()
    print(f"  BEFORE: {old_a[0]:>7} trades | WR={old_a[1]:>5.1f}% | "
          f"AvgPnl={old_a[2]:>7.4f}% | Total=${old_a[3]:>12.2f}")
    print(f"  AFTER:  {new_a[0]:>7} trades | WR={new_a[1]:>5.1f}% | "
          f"AvgPnl={new_a[2]:>7.4f}% | Total=${new_a[3]:>12.2f}")

print("\nDone!")
