"""
Analyze regime performance for each strategy to find profitable windows.
Outputs recommendations for regime filters.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine, text
from src.config import settings


def main():
    engine = create_engine(settings.sync_db_url)

    with engine.connect() as conn:
        # Overall strategy summary
        print("=" * 90)
        print("OVERALL STRATEGY PERFORMANCE")
        print("=" * 90)
        rows = conn.execute(text("""
            SELECT
                strategy_combo,
                symbol,
                COUNT(*) AS trades,
                ROUND(AVG(win_loss::int) * 100, 1) AS wr,
                ROUND(AVG(pnl_pct) * 100, 4) AS avg_pnl,
                ROUND(SUM(pnl_usd), 2) AS total_pnl
            FROM trades_log
            WHERE is_backtest = TRUE AND exit_time IS NOT NULL
            GROUP BY strategy_combo, symbol
            HAVING COUNT(*) >= 10
            ORDER BY AVG(pnl_pct) DESC
        """)).fetchall()

        print(f"{'Combo':<55} {'Sym':<10} {'Trades':>7} {'WR%':>6} {'AvgPnl%':>8} {'TotalPnL':>10}")
        print("-" * 90)
        for r in rows:
            combo_str = str(r[0])[:54]
            print(f"{combo_str:<55} {r[1]:<10} {r[2]:>7} {r[3]:>5.1f}% {r[4]:>7.4f}% ${r[5]:>9.2f}")

        # Full regime breakdown for each strategy combo
        print("\n" + "=" * 90)
        print("REGIME BREAKDOWN BY STRATEGY")
        print("=" * 90)

        combos = conn.execute(text("""
            SELECT DISTINCT strategy_combo FROM trades_log
            WHERE is_backtest = TRUE AND exit_time IS NOT NULL
            GROUP BY strategy_combo HAVING COUNT(*) >= 20
            ORDER BY strategy_combo
        """)).fetchall()

        for (combo,) in combos:
            print(f"\n--- {combo} ---")
            regime_rows = conn.execute(text("""
                SELECT
                    regime_volatility,
                    regime_funding,
                    regime_time_of_day,
                    COUNT(*) AS trades,
                    ROUND(AVG(win_loss::int) * 100, 1) AS wr,
                    ROUND(AVG(pnl_pct) * 100, 4) AS avg_pnl,
                    ROUND(SUM(pnl_usd), 2) AS total_pnl
                FROM trades_log
                WHERE is_backtest = TRUE
                  AND exit_time IS NOT NULL
                  AND strategy_combo = :combo
                GROUP BY regime_volatility, regime_funding, regime_time_of_day
                HAVING COUNT(*) >= 5
                ORDER BY AVG(pnl_pct) DESC
            """), {"combo": combo}).fetchall()

            if not regime_rows:
                print("  (insufficient data)")
                continue

            print(f"  {'Vol':<8} {'Funding':<10} {'Session':<12} {'Trades':>7} {'WR%':>6} {'AvgPnl%':>8} {'TotalPnL':>10}")
            print(f"  {'-'*70}")
            for r in regime_rows:
                marker = ""
                if r[4] and float(r[4]) >= 0.65:
                    marker = " <<<< GREEN"
                elif r[4] and float(r[4]) < 0.40:
                    marker = " <<<< AVOID"
                print(f"  {r[0] or '?':<8} {r[1] or '?':<10} {r[2] or '?':<12} "
                      f"{r[3]:>7} {r[4]:>5.1f}% {r[5]:>7.4f}% ${r[6]:>9.2f}{marker}")

        # Direction breakdown
        print("\n" + "=" * 90)
        print("DIRECTION BREAKDOWN (Long vs Short)")
        print("=" * 90)
        dir_rows = conn.execute(text("""
            SELECT
                strategy_combo,
                direction,
                COUNT(*) AS trades,
                ROUND(AVG(win_loss::int) * 100, 1) AS wr,
                ROUND(AVG(pnl_pct) * 100, 4) AS avg_pnl,
                ROUND(SUM(pnl_usd), 2) AS total_pnl
            FROM trades_log
            WHERE is_backtest = TRUE AND exit_time IS NOT NULL
            GROUP BY strategy_combo, direction
            HAVING COUNT(*) >= 10
            ORDER BY strategy_combo, direction
        """)).fetchall()

        print(f"{'Combo':<55} {'Dir':<6} {'Trades':>7} {'WR%':>6} {'AvgPnl%':>8} {'TotalPnL':>10}")
        print("-" * 90)
        for r in dir_rows:
            combo_str = str(r[0])[:54]
            print(f"{combo_str:<55} {r[1]:<6} {r[2]:>7} {r[3]:>5.1f}% {r[4]:>7.4f}% ${r[5]:>9.2f}")

        # Exit reason breakdown
        print("\n" + "=" * 90)
        print("EXIT REASON BREAKDOWN")
        print("=" * 90)
        exit_rows = conn.execute(text("""
            SELECT
                strategy_combo,
                exit_reason,
                COUNT(*) AS trades,
                ROUND(AVG(win_loss::int) * 100, 1) AS wr,
                ROUND(AVG(pnl_pct) * 100, 4) AS avg_pnl
            FROM trades_log
            WHERE is_backtest = TRUE AND exit_time IS NOT NULL
            GROUP BY strategy_combo, exit_reason
            HAVING COUNT(*) >= 10
            ORDER BY strategy_combo, exit_reason
        """)).fetchall()

        print(f"{'Combo':<55} {'Exit':<10} {'Trades':>7} {'WR%':>6} {'AvgPnl%':>8}")
        print("-" * 90)
        for r in exit_rows:
            combo_str = str(r[0])[:54]
            print(f"{combo_str:<55} {r[1]:<10} {r[2]:>7} {r[3]:>5.1f}% {r[4]:>7.4f}%")

        # Volatility regime summary
        print("\n" + "=" * 90)
        print("VOLATILITY REGIME SUMMARY (all strategies)")
        print("=" * 90)
        vol_rows = conn.execute(text("""
            SELECT
                regime_volatility,
                COUNT(*) AS trades,
                ROUND(AVG(win_loss::int) * 100, 1) AS wr,
                ROUND(AVG(pnl_pct) * 100, 4) AS avg_pnl,
                ROUND(SUM(pnl_usd), 2) AS total_pnl
            FROM trades_log
            WHERE is_backtest = TRUE AND exit_time IS NOT NULL
            GROUP BY regime_volatility
            ORDER BY regime_volatility
        """)).fetchall()

        for r in vol_rows:
            print(f"  {r[0] or '?':<10} {r[1]:>8} trades  WR={r[2]:>5.1f}%  "
                  f"AvgPnl={r[3]:>7.4f}%  TotalPnL=${r[4]:>10.2f}")

        # Funding regime summary
        print("\n" + "=" * 90)
        print("FUNDING REGIME SUMMARY (all strategies)")
        print("=" * 90)
        fund_rows = conn.execute(text("""
            SELECT
                regime_funding,
                COUNT(*) AS trades,
                ROUND(AVG(win_loss::int) * 100, 1) AS wr,
                ROUND(AVG(pnl_pct) * 100, 4) AS avg_pnl,
                ROUND(SUM(pnl_usd), 2) AS total_pnl
            FROM trades_log
            WHERE is_backtest = TRUE AND exit_time IS NOT NULL
            GROUP BY regime_funding
            ORDER BY regime_funding
        """)).fetchall()

        for r in fund_rows:
            print(f"  {r[0] or '?':<10} {r[1]:>8} trades  WR={r[2]:>5.1f}%  "
                  f"AvgPnl={r[3]:>7.4f}%  TotalPnL=${r[4]:>10.2f}")

        # Session regime summary
        print("\n" + "=" * 90)
        print("SESSION REGIME SUMMARY (all strategies)")
        print("=" * 90)
        sess_rows = conn.execute(text("""
            SELECT
                regime_time_of_day,
                COUNT(*) AS trades,
                ROUND(AVG(win_loss::int) * 100, 1) AS wr,
                ROUND(AVG(pnl_pct) * 100, 4) AS avg_pnl,
                ROUND(SUM(pnl_usd), 2) AS total_pnl
            FROM trades_log
            WHERE is_backtest = TRUE AND exit_time IS NOT NULL
            GROUP BY regime_time_of_day
            ORDER BY regime_time_of_day
        """)).fetchall()

        for r in sess_rows:
            print(f"  {r[0] or '?':<12} {r[1]:>8} trades  WR={r[2]:>5.1f}%  "
                  f"AvgPnl={r[3]:>7.4f}%  TotalPnL=${r[4]:>10.2f}")


if __name__ == "__main__":
    main()
