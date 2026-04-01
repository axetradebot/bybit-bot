"""Compare before/after regime filtering results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text
from src.config import settings

engine = create_engine(settings.sync_db_url)

with engine.connect() as conn:
    # Check if pre-regime table exists
    exists = conn.execute(text(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
        "WHERE table_name = 'trades_log_pre_regime')"
    )).scalar()

    if not exists:
        print("No pre-regime data found. Cannot compare.")
        sys.exit(1)

    print("=" * 100)
    print("BEFORE/AFTER COMPARISON — Regime Filters + MACD(5,13,3) Fix")
    print("=" * 100)

    # Per-strategy comparison
    combos = conn.execute(text("""
        SELECT DISTINCT strategy_combo FROM (
            SELECT strategy_combo FROM trades_log_pre_regime
            UNION
            SELECT strategy_combo FROM trades_log WHERE is_backtest = TRUE
        ) x ORDER BY strategy_combo
    """)).fetchall()

    print(f"\n{'Combo':<42} {'':>4}| {'Trades':>7} {'WR%':>6} {'AvgPnl%':>8} {'TotalPnL$':>11}")
    print("-" * 100)

    for (combo,) in combos:
        old = conn.execute(text("""
            SELECT COUNT(*),
                   ROUND(AVG(win_loss::int)*100,1),
                   ROUND(AVG(pnl_pct)*100,4),
                   ROUND(SUM(pnl_usd),2)
            FROM trades_log_pre_regime
            WHERE strategy_combo = :c
        """), {"c": combo}).fetchone()

        new = conn.execute(text("""
            SELECT COUNT(*),
                   ROUND(AVG(win_loss::int)*100,1),
                   ROUND(AVG(pnl_pct)*100,4),
                   ROUND(SUM(pnl_usd),2)
            FROM trades_log WHERE is_backtest = TRUE AND strategy_combo = :c
        """), {"c": combo}).fetchone()

        combo_short = str(combo)[:41]
        if old[0] > 0:
            print(f"{combo_short:<42} OLD | {old[0]:>7} {old[1]:>5.1f}% {old[2]:>7.4f}% ${old[3]:>10.2f}")
        if new[0] > 0:
            print(f"{'':<42} NEW | {new[0]:>7} {new[1]:>5.1f}% {new[2]:>7.4f}% ${new[3]:>10.2f}")
        elif new[0] == 0:
            print(f"{'':<42} NEW | {'0 trades':<40}")

        if old[0] > 0 and new[0] > 0:
            trade_delta = new[0] - old[0]
            wr_delta = float(new[1]) - float(old[1])
            pnl_delta = float(new[2]) - float(old[2])
            pnl_usd_delta = float(new[3]) - float(old[3])
            direction = "+" if wr_delta >= 0 else ""
            print(f"{'':<42}   D | {trade_delta:>+7} {direction}{wr_delta:>4.1f}% {pnl_delta:>+7.4f}% ${pnl_usd_delta:>+10.2f}")
        print()

    # Aggregate
    print("=" * 100)
    print("AGGREGATE COMPARISON")
    print("=" * 100)

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

    trade_chg = (new_a[0] - old_a[0]) / old_a[0] * 100 if old_a[0] > 0 else 0
    wr_chg = float(new_a[1]) - float(old_a[1])
    pnl_chg = float(new_a[2]) - float(old_a[2])
    print(f"\n  Trades: {trade_chg:+.1f}%")
    print(f"  WR:     {wr_chg:+.1f}pp")
    print(f"  AvgPnl: {pnl_chg:+.4f}pp")
