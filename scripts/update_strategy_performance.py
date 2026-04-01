"""
Aggregate trades_log into strategy_performance table.
Run after backtests to make regime_adaptive work.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config import settings


def main():
    engine = create_engine(settings.sync_db_url)

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM strategy_performance WHERE 1=1"))

        rows = conn.execute(text("""
            SELECT
                strategy_combo,
                symbol,
                COUNT(*) AS total_trades,
                AVG(win_loss::int) AS win_rate,
                AVG(pnl_pct) AS avg_pnl_pct,
                regime_funding
            FROM trades_log
            WHERE is_backtest = TRUE AND exit_time IS NOT NULL
            GROUP BY strategy_combo, symbol, regime_funding
            HAVING COUNT(*) >= 5
        """)).fetchall()

        for r in rows:
            combo, symbol, total, wr, avg_pnl, fund = r
            wr = float(wr) if wr else 0.0
            avg_pnl = float(avg_pnl) if avg_pnl else 0.0
            avg_win = max(avg_pnl, 0.0)
            avg_loss = abs(min(avg_pnl, 0.0))
            expectancy = wr * avg_win - (1 - wr) * avg_loss

            pnls = conn.execute(text("""
                SELECT pnl_pct FROM trades_log
                WHERE strategy_combo = :combo AND symbol = :sym
                  AND is_backtest = TRUE AND regime_funding = :fund
                ORDER BY entry_time
            """), {"combo": combo, "sym": symbol, "fund": fund}).fetchall()

            pnl_arr = np.array([float(p[0]) for p in pnls if p[0] is not None])
            if len(pnl_arr) >= 2:
                std = pnl_arr.std(ddof=1)
                sharpe = float(pnl_arr.mean() / std * math.sqrt(252)) if std > 0 else 0.0
                cum = np.cumsum(pnl_arr)
                max_dd = float((cum - np.maximum.accumulate(cum)).min())
            else:
                sharpe = 0.0
                max_dd = 0.0

            conn.execute(text("""
                INSERT INTO strategy_performance
                    (strategy_combo, symbol, timeframe, total_trades,
                     win_rate, avg_pnl_pct, expectancy, sharpe,
                     max_drawdown, regime_funding, last_updated)
                VALUES
                    (:combo, :sym, '5m', :total,
                     :wr, :avg_pnl, :exp, :sharpe,
                     :dd, :fund, NOW())
                ON CONFLICT ON CONSTRAINT uq_strategy_performance_combo
                DO UPDATE SET
                    total_trades = EXCLUDED.total_trades,
                    win_rate = EXCLUDED.win_rate,
                    avg_pnl_pct = EXCLUDED.avg_pnl_pct,
                    expectancy = EXCLUDED.expectancy,
                    sharpe = EXCLUDED.sharpe,
                    max_drawdown = EXCLUDED.max_drawdown,
                    last_updated = NOW()
            """), {
                "combo": combo, "sym": symbol, "total": total,
                "wr": wr, "avg_pnl": avg_pnl, "exp": expectancy,
                "sharpe": sharpe, "dd": max_dd, "fund": fund,
            })

        print(f"Updated {len(rows)} strategy_performance rows")

    with engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM strategy_performance")).scalar()
        print(f"Total rows in strategy_performance: {count}")

        top = conn.execute(text("""
            SELECT strategy_combo, symbol,
                   total_trades, ROUND(win_rate::numeric, 3),
                   ROUND(expectancy::numeric, 5),
                   ROUND(sharpe::numeric, 2)
            FROM strategy_performance
            ORDER BY expectancy DESC NULLS LAST
            LIMIT 10
        """)).fetchall()

        print("\nTop strategies:")
        for r in top:
            print(f"  {r[0]} / {r[1]}: {r[2]} trades, WR={r[3]}, E={r[4]}, S={r[5]}")


if __name__ == "__main__":
    main()
