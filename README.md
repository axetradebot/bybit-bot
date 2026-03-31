# Bybit Futures Trading Bot

Automated USDT-Margined Perpetual Futures trading bot for Bybit. Combines
multi-timeframe technical analysis, regime-aware strategies, and a full risk
management layer with a shadow-database architecture that makes backtest and
live trades analytically identical.

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  Bybit Exchange (testnet / mainnet)                  │
│    WebSocket: kline.5, tickers, execution            │
│    REST (CCXT): orders, positions, account           │
└────────────┬──────────────────────────┬──────────────┘
             │                          │
   ┌─────────▼─────────┐    ┌──────────▼──────────┐
   │ WebSocketListener  │    │   OrderManager      │
   │ (pybit)            │───▶│   (ccxt)            │
   │ bar_buffer[200]    │    │   limit entries     │
   │ indicator compute  │    │   stop-market SL    │
   │ strategy dispatch  │    │   limit TP          │
   └─────────┬──────────┘    └─────────────────────┘
             │
   ┌─────────▼──────────┐
   │ RiskManager         │  8 sequential gates:
   │                     │  daily_loss → session → max_pos →
   │                     │  funding → leverage → size →
   │                     │  rr → liq_cluster
   └─────────┬──────────┘
             │
   ┌─────────▼──────────┐
   │ Strategies (6)      │  bb_squeeze, rsi_divergence,
   │                     │  vwap_reversion, multitf_scalp,
   │                     │  volume_delta_liq, regime_adaptive
   └─────────────────────┘

   ┌─────────────────────┐    ┌─────────────────────┐
   │ TimescaleDB          │    │ Streamlit Dashboard  │
   │ (shadow database)    │◀───│ Strategy Explorer    │
   │ candles, indicators, │    │ Regime Analysis      │
   │ trades_log           │    │ Live Monitor         │
   └─────────────────────┘    │ Data Quality         │
                               └─────────────────────┘
```

## Prerequisites

- **Python 3.12+**
- **Docker** and **docker-compose** (for TimescaleDB)
- **Tardis.dev** API key (historical data)
- **Bybit** API key/secret (live trading; testnet recommended)

## Quick Start (Local Development)

```bash
# 1. Clone & enter project
cd bybit_futures_bot

# 2. Copy and edit environment
cp .env.example .env
#    → Set DB_USER, DB_PASSWORD, TARDIS_API_KEY
#    → Set BYBIT_API_KEY, BYBIT_API_SECRET for live trading
#    → Keep BYBIT_TESTNET=true until verified

# 3. Start infrastructure
docker-compose up -d

# 4. Install Python dependencies
pip install -r requirements.txt

# 5. Initialise database (migrations + hypertables)
python scripts/init_db.py

# 6. Download historical data
python src/data/download_historical.py --from 2021-01-01

# 7. Compute indicators
python src/indicators/compute_all.py --symbol BTCUSDT --from 2021-01-01
python src/indicators/compute_all.py --symbol ETHUSDT --from 2021-01-01
python src/indicators/compute_all.py --symbol SOLUSDT --from 2021-01-01

# 8. Run a backtest
python src/backtest/run_backtest.py \
    --strategy bb_squeeze \
    --symbol BTCUSDT \
    --from 2021-01-01 \
    --to 2024-12-31 \
    --leverage 10 \
    --risk-pct 0.02

# 9. Launch analytics dashboard
streamlit run src/analytics/streamlit_dashboard.py

# 10. Start live bot (testnet)
python src/live/websocket_listener.py
```

## Database Schema

| Table                  | Description                              |
|------------------------|------------------------------------------|
| `candles_5m`           | 5-minute OHLCV + buy/sell volume         |
| `candles_1m`           | 1-minute OHLCV (raw granularity)         |
| `indicators_5m`        | Technical indicators on 5m timeframe     |
| `indicators_15m`       | Technical indicators on 15m timeframe    |
| `funding_history`      | 8-hourly funding rates per symbol        |
| `liquidations`         | Liquidation events                       |
| `raw_trades`           | Tick-level trade data                    |
| `trades_log`           | Backtest + live trade records (unified)  |
| `strategy_performance` | Aggregated strategy combo performance    |

All time-series tables are TimescaleDB hypertables with automatic compression.

## Strategies

| Name                      | Logic                                                    |
|---------------------------|----------------------------------------------------------|
| `bb_squeeze`              | Bollinger Band squeeze release + EMA trend + MACD + VWAP |
| `rsi_divergence`          | RSI divergence + SuperTrend flip + EMA slope             |
| `vwap_reversion`          | Mean reversion to VWAP in ranging markets                |
| `multitf_scalp`           | 15m trend alignment + 5m pullback entry                  |
| `volume_delta_liq`        | Order flow breakout near liquidation clusters            |
| `regime_adaptive`         | Dynamically selects best combo from strategy_performance |

## Risk Management

Every signal passes through 8 sequential gates before order placement:

1. **Daily Loss Gate** — halts trading if daily drawdown exceeds 5%
2. **Session Open Gate** — blocks trades within 5 min of session opens
3. **Max Positions Gate** — limits to 3 concurrent positions, 1 per symbol
4. **Funding Gate** — blocks adverse funding rate conditions
5. **Leverage Gate** — dynamically caps leverage based on historical win rate
6. **Position Size Gate** — sizes to 1% risk; blocks if liquidation too close
7. **R:R Gate** — requires risk-reward ratio >= 1.5
8. **Liquidation Cluster Gate** — reduces size 50% near extreme liquidation zones

## Configuration

All settings are loaded from `.env`:

| Variable            | Description                          | Default          |
|---------------------|--------------------------------------|------------------|
| `DB_HOST`           | PostgreSQL host                      | `localhost`      |
| `DB_PORT`           | PostgreSQL port                      | `5432`           |
| `DB_NAME`           | Database name                        | `bybit_shadow`   |
| `DB_USER`           | Database user                        | (required)       |
| `DB_PASSWORD`       | Database password                    | (required)       |
| `SYMBOLS`           | Traded symbols (JSON list)           | BTC, ETH, SOL    |
| `TARDIS_API_KEY`    | Tardis.dev API key                   | (required)       |
| `BYBIT_API_KEY`     | Bybit API key                        | (optional)       |
| `BYBIT_API_SECRET`  | Bybit API secret                     | (optional)       |
| `BYBIT_TESTNET`     | Use Bybit testnet                    | `true`           |

## Backtesting

```bash
python src/backtest/run_backtest.py \
    --strategy bb_squeeze \
    --symbol BTCUSDT \
    --from 2023-01-01 \
    --to 2024-12-31 \
    --leverage 10 \
    --risk-pct 0.02
```

Available strategies: `bb_squeeze`, `rsi_divergence`, `vwap_reversion`,
`multitf_scalp`, `volume_delta_liq`.

Backtest trades are written to `trades_log` with `is_backtest = TRUE` and are
visible in the dashboard alongside live trades.

## Analytics Dashboard

```bash
streamlit run src/analytics/streamlit_dashboard.py
```

Four pages:

1. **Strategy Explorer** — filter by symbol, date, regime; sortable
   performance table, equity curve, win-rate heatmap
2. **Regime Analysis** — win rate breakdown by funding × volatility × session;
   green/red highlighting for strong/weak regimes
3. **Live Monitor** — open positions, today's PnL, recent signals, funding
4. **Data Quality** — row counts, gap detection, NULL checks, sync timestamps

## Production Deployment (AWS EC2)

```bash
# On a fresh Ubuntu 22.04 EC2 (t3.medium+, 100 GB gp3)
sudo bash deployment/aws_ec2_setup.sh

# Edit .env
nano /opt/bybit_bot/.env

# Start database
sudo bash deployment/timescaledb_setup.sh

# Start bot
sudo systemctl start bybit-bot
sudo systemctl status bybit-bot

# Health check
curl http://localhost:8000/health
```

The bot runs as a systemd service with automatic restart on failure.

## Monitoring

The FastAPI health API runs on port 8000:

| Endpoint            | Method | Description                       |
|---------------------|--------|-----------------------------------|
| `/health`           | GET    | `{"status": "ok"}` for watchdogs  |
| `/status`           | GET    | Bot state, uptime, WS connection  |
| `/positions`        | GET    | Open positions with current PnL   |
| `/pnl`              | GET    | Today / week / all-time PnL       |
| `/top-strategies`   | GET    | Top 10 combos by expectancy       |
| `/blocked-signals`  | GET    | Last 20 blocked signals + reasons |
| `/inject-signal`    | POST   | Push synthetic signal for testing  |

## Safety

- **Testnet by default**: `BYBIT_TESTNET=true` in `.env`
- **Hard assertions**: `assert settings.bybit_testnet` before any order placement
- **No market orders for entries**: always aggressive limit orders
- **Risk gates**: 8 sequential checks before every trade
- Only switch `BYBIT_TESTNET=false` after reviewing 50+ testnet trades in the dashboard
