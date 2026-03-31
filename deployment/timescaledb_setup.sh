#!/bin/bash
# =============================================================
# Bybit Futures Bot — TimescaleDB Setup
# Run AFTER aws_ec2_setup.sh to start TimescaleDB and init schema
# =============================================================
set -euo pipefail

cd /opt/bybit_bot

echo "==> Starting TimescaleDB"
docker-compose -f docker-compose.prod.yml up -d timescaledb

echo "==> Waiting for TimescaleDB to become healthy (up to 60s)..."
for i in $(seq 1 12); do
    if docker exec bybit_timescaledb pg_isready -U bybit -d bybit_shadow > /dev/null 2>&1; then
        echo "    TimescaleDB is ready."
        break
    fi
    echo "    Waiting... ($i/12)"
    sleep 5
done

echo "==> Activating virtual environment"
source /opt/bybit_bot/.venv/bin/activate

echo "==> Running database initialisation (migrations + hypertables)"
python scripts/init_db.py

echo "============================================"
echo "  TimescaleDB ready."
echo ""
echo "  Next steps:"
echo "    1. Download historical data:"
echo "       python src/data/download_historical.py --from 2021-01-01"
echo "    2. Compute indicators:"
echo "       python src/indicators/compute_all.py --symbol BTCUSDT --from 2021-01-01"
echo "    3. Start the bot:"
echo "       systemctl start bybit-bot"
echo "============================================"
