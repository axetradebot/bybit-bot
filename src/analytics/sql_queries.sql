-- ============================================================
-- Bybit Futures Bot — Analytics Queries
-- Run these against the bybit_shadow database.
-- ============================================================


-- 1. Find best overall combos (minimum viable filter)
SELECT
    strategy_combo,
    symbol,
    total_trades,
    ROUND(win_rate * 100, 1)           AS win_rate_pct,
    ROUND(expectancy, 4)               AS expectancy,
    ROUND(sharpe, 2)                   AS sharpe,
    ROUND(max_drawdown * 100, 1)       AS max_dd_pct
FROM strategy_performance
WHERE total_trades  >= 100
  AND win_rate      >= 0.55
  AND expectancy    > 0
ORDER BY expectancy DESC
LIMIT 20;


-- 2. Regime breakdown for a specific combo
SELECT
    regime_volatility,
    regime_funding,
    regime_time_of_day,
    COUNT(*)                                    AS trades,
    ROUND(AVG(win_loss::int) * 100, 1)          AS win_rate_pct,
    ROUND(AVG(pnl_pct) * 100, 4)               AS avg_pnl_pct,
    ROUND(SUM(pnl_usd), 2)                     AS total_pnl_usd
FROM trades_log
WHERE strategy_combo @> ARRAY['bb_squeeze', 'ema_trend']
  AND symbol       = 'BTCUSDT'
  AND is_backtest  = TRUE
GROUP BY regime_volatility, regime_funding, regime_time_of_day
ORDER BY win_rate_pct DESC;


-- 3. Find the single best regime for each combo
SELECT DISTINCT ON (strategy_combo, symbol)
    strategy_combo,
    symbol,
    regime_volatility,
    regime_funding,
    regime_time_of_day,
    COUNT(*)                                    AS trades,
    ROUND(AVG(win_loss::int) * 100, 1)          AS win_rate_pct
FROM trades_log
WHERE is_backtest = TRUE
GROUP BY strategy_combo, symbol,
         regime_volatility, regime_funding, regime_time_of_day
HAVING COUNT(*) >= 30
ORDER BY strategy_combo, symbol, AVG(win_loss::int) DESC;


-- 4. Funding gate impact — are we blocking too many good trades?
SELECT
    exit_reason,
    COUNT(*)                                    AS blocked,
    AVG(CASE WHEN exit_reason = 'funding_gate'
        THEN (indicators_snapshot->>'rsi_14')::float END) AS avg_rsi_at_block
FROM trades_log
WHERE win_loss IS NULL
  AND symbol   = 'BTCUSDT'
GROUP BY exit_reason
ORDER BY blocked DESC;


-- 5. Leverage optimisation per combo
SELECT
    leverage,
    COUNT(*)                                    AS trades,
    ROUND(AVG(win_loss::int) * 100, 1)          AS win_rate_pct,
    ROUND(SUM(pnl_usd), 2)                     AS total_pnl_usd,
    ROUND(MIN(pnl_pct) * 100, 2)               AS worst_trade_pct
FROM trades_log
WHERE strategy_combo @> ARRAY['rsi_div', 'supertrend_flip']
  AND is_backtest    = TRUE
GROUP BY leverage
ORDER BY total_pnl_usd DESC;


-- 6. Week-over-week degradation check
--    Run monthly to detect if a strategy is losing edge
SELECT
    DATE_TRUNC('week', entry_time)              AS week,
    COUNT(*)                                    AS trades,
    ROUND(AVG(win_loss::int) * 100, 1)          AS win_rate_pct,
    ROUND(AVG(pnl_pct) * 100, 4)               AS avg_pnl_pct
FROM trades_log
WHERE strategy_combo @> ARRAY['bb_squeeze', 'ema_trend']
  AND symbol       = 'BTCUSDT'
  AND is_backtest  = TRUE
GROUP BY week
ORDER BY week;
