-- TimescaleDB extension (already loaded via shared_preload_libraries)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Convert tables to TimescaleDB hypertables
SELECT create_hypertable('candles_5m', 'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);

SELECT create_hypertable('candles_1m', 'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE);

SELECT create_hypertable('raw_trades', 'timestamp',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE);

SELECT create_hypertable('funding_history', 'timestamp',
    chunk_time_interval => INTERVAL '3 months',
    if_not_exists => TRUE);

SELECT create_hypertable('liquidations', 'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);

SELECT create_hypertable('indicators_5m', 'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);

SELECT create_hypertable('indicators_15m', 'timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_candles5m_symbol_ts
    ON candles_5m (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_indicators5m_symbol_ts
    ON indicators_5m (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_trades_log_combo
    ON trades_log USING GIN (strategy_combo);

CREATE INDEX IF NOT EXISTS idx_trades_log_symbol_entry
    ON trades_log (symbol, entry_time DESC);

CREATE INDEX IF NOT EXISTS idx_trades_log_is_backtest
    ON trades_log (is_backtest, symbol);

CREATE INDEX IF NOT EXISTS idx_funding_symbol_ts
    ON funding_history (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_liquidations_symbol_ts
    ON liquidations (symbol, timestamp DESC);

-- Enable compression on hypertables (saves ~80% storage)
ALTER TABLE candles_5m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('candles_5m', INTERVAL '7 days');

ALTER TABLE candles_1m SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('candles_1m', INTERVAL '7 days');

ALTER TABLE raw_trades SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('raw_trades', INTERVAL '3 days');
