"""
Realistic equity simulation with $1,000 starting balance.

The backtest simulator already charges maker fees and uses limit order fills.
This simulation walks trades chronologically with compounding equity,
allowing one position per strategy-symbol pair (as the live bot would).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime
from collections import defaultdict
from sqlalchemy import create_engine, text
from src.config import settings


def main():
    engine = create_engine(settings.sync_db_url)
    START_EQUITY = 1_000.0
    RISK_PCT = 0.02
    LEVERAGE = 10

    with engine.connect() as conn:
        trades = conn.execute(text("""
            SELECT
                strategy_combo, symbol, direction,
                entry_time, exit_time,
                entry_price, exit_price,
                stop_loss, take_profit,
                pnl_pct, win_loss, exit_reason
            FROM trades_log
            WHERE is_backtest = TRUE AND exit_time IS NOT NULL
            ORDER BY entry_time, exit_time
        """)).fetchall()

    if not trades:
        print("No backtest trades found!")
        return

    # Group trades by strategy-symbol pair (each pair is independent)
    pair_trades = defaultdict(list)
    for t in trades:
        key = (str(t[0]), t[1])  # (strategy_combo, symbol)
        pair_trades[key].append(t)

    n_pairs = len(pair_trades)
    equity = START_EQUITY
    peak = equity
    max_dd_pct = 0.0

    monthly = defaultdict(lambda: {"start_eq": None, "pnl": 0.0, "trades": 0, "wins": 0})
    yearly = defaultdict(lambda: {"start_eq": None, "pnl": 0.0, "trades": 0, "wins": 0})
    strategy_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
    total_wins = 0
    total_losses = 0
    biggest_win = 0.0
    biggest_loss = 0.0
    max_consec_loss = 0
    consec_loss = 0

    # Merge all trades into one timeline, respecting that each pair's
    # trades are already non-overlapping (simulator ensures this).
    # Capital is shared — each trade risks RISK_PCT of current equity.
    all_trades_sorted = sorted(trades, key=lambda t: t[4])  # sort by exit_time

    for t in all_trades_sorted:
        (combo, symbol, direction,
         entry_time, exit_time,
         entry_price, exit_price,
         stop_loss, take_profit,
         pnl_pct, win_loss, exit_reason) = t

        pnl_pct = float(pnl_pct)
        entry_price = float(entry_price)
        stop_loss = float(stop_loss)
        exit_time = exit_time if isinstance(exit_time, datetime) else datetime.fromisoformat(str(exit_time))

        if equity <= 10:  # account effectively blown
            break

        # Position size based on current equity (fees already in pnl_pct)
        if direction == "long":
            sl_dist_pct = abs(entry_price - stop_loss) / entry_price
        else:
            sl_dist_pct = abs(stop_loss - entry_price) / entry_price
        sl_dist_pct = max(sl_dist_pct, 0.001)

        risk_usd = equity * RISK_PCT
        position_size = min(risk_usd / sl_dist_pct, equity * LEVERAGE)

        # pnl_pct already includes fees from the simulator
        trade_pnl = position_size * pnl_pct
        equity += trade_pnl

        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0
        max_dd_pct = max(max_dd_pct, dd)

        if win_loss:
            total_wins += 1
            consec_loss = 0
        else:
            total_losses += 1
            consec_loss += 1
            max_consec_loss = max(max_consec_loss, consec_loss)

        biggest_win = max(biggest_win, pnl_pct)
        biggest_loss = min(biggest_loss, pnl_pct)

        mk = exit_time.strftime("%Y-%m")
        if monthly[mk]["start_eq"] is None:
            monthly[mk]["start_eq"] = equity - trade_pnl
        monthly[mk]["pnl"] += trade_pnl
        monthly[mk]["trades"] += 1
        if win_loss:
            monthly[mk]["wins"] += 1

        yk = exit_time.strftime("%Y")
        if yearly[yk]["start_eq"] is None:
            yearly[yk]["start_eq"] = equity - trade_pnl
        yearly[yk]["pnl"] += trade_pnl
        yearly[yk]["trades"] += 1
        if win_loss:
            yearly[yk]["wins"] += 1

        combo_key = str(combo)
        strategy_stats[combo_key]["trades"] += 1
        strategy_stats[combo_key]["pnl"] += trade_pnl
        if win_loss:
            strategy_stats[combo_key]["wins"] += 1

    total_trades = total_wins + total_losses
    first = trades[0][3]
    last = trades[-1][4]
    days = (last - first).days
    years = days / 365.25
    total_return = (equity - START_EQUITY) / START_EQUITY * 100
    cagr = ((max(equity, 0.01) / START_EQUITY) ** (1 / years) - 1) * 100 if years > 0 else 0

    print("=" * 80)
    print(f"EQUITY SIMULATION (fees already included in backtest PnL)")
    print(f"Start: ${START_EQUITY:,.0f} | Risk: {RISK_PCT:.0%}/trade | "
          f"Leverage: {LEVERAGE}x | Strategy-Symbol pairs: {n_pairs}")
    print(f"Period: {first.strftime('%Y-%m-%d')} to {last.strftime('%Y-%m-%d')} "
          f"({days} days / {years:.1f} years)")
    print("=" * 80)

    print(f"\n  Starting Equity:    ${START_EQUITY:>14,.2f}")
    print(f"  Final Equity:       ${equity:>14,.2f}")
    print(f"  Total Return:       {total_return:>14.1f}%")
    print(f"  CAGR:               {cagr:>14.1f}%")

    print(f"\n  Total Trades:       {total_trades:>14,}")
    print(f"  Wins / Losses:      {total_wins:>7,} / {total_losses:>5,}")
    print(f"  Win Rate:           {total_wins/total_trades*100 if total_trades else 0:>14.1f}%")

    print(f"\n  Max Drawdown:       {max_dd_pct*100:>14.1f}%")
    print(f"  Peak Equity:        ${peak:>14,.2f}")

    print(f"\n  Biggest Win:        {biggest_win*100:>14.2f}%")
    print(f"  Biggest Loss:       {biggest_loss*100:>14.2f}%")
    print(f"  Max Losing Streak:  {max_consec_loss:>14}")

    if max_dd_pct > 0 and years > 0:
        calmar = cagr / (max_dd_pct * 100)
        print(f"  Calmar Ratio:       {calmar:>14.2f}")

    # Yearly
    print(f"\n{'YEARLY BREAKDOWN':=^80}")
    print(f"  {'Year':<6} {'Start$':>12} {'PnL$':>12} {'End$':>12} {'Ret%':>8} {'Trades':>7} {'WR%':>6}")
    print(f"  {'-'*72}")
    for yr in sorted(yearly.keys()):
        y = yearly[yr]
        s = y["start_eq"]
        e = s + y["pnl"]
        ret = y["pnl"] / s * 100 if s > 0 else 0
        wr = y["wins"] / y["trades"] * 100 if y["trades"] > 0 else 0
        print(f"  {yr:<6} ${s:>11,.2f} ${y['pnl']:>+11,.2f} ${e:>11,.2f} "
              f"{ret:>+7.1f}% {y['trades']:>7} {wr:>5.1f}%")

    # Monthly
    print(f"\n{'MONTHLY BREAKDOWN':=^80}")
    print(f"  {'Month':<8} {'Start$':>10} {'PnL$':>10} {'End$':>10} {'Ret%':>8} {'N':>5} {'WR%':>6}")
    print(f"  {'-'*62}")
    for mo in sorted(monthly.keys()):
        m = monthly[mo]
        s = m["start_eq"]
        e = s + m["pnl"]
        ret = m["pnl"] / s * 100 if s > 0 else 0
        wr = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
        flag = " <<<" if ret < -5 else (" ***" if ret > 5 else "")
        print(f"  {mo:<8} ${s:>9,.2f} ${m['pnl']:>+9,.2f} ${e:>9,.2f} "
              f"{ret:>+7.1f}% {m['trades']:>5} {wr:>5.1f}%{flag}")

    # Strategy contribution
    print(f"\n{'STRATEGY CONTRIBUTION':=^80}")
    print(f"  {'Strategy':<55} {'N':>5} {'WR%':>6} {'PnL$':>10}")
    print(f"  {'-'*78}")
    for c in sorted(strategy_stats.keys(),
                    key=lambda x: strategy_stats[x]["pnl"], reverse=True):
        s = strategy_stats[c]
        wr = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
        print(f"  {c[:54]:<55} {s['trades']:>5} {wr:>5.1f}% ${s['pnl']:>+9,.2f}")


if __name__ == "__main__":
    main()
