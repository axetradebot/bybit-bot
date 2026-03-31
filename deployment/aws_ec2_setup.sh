#!/bin/bash
# =============================================================
# Bybit Futures Bot — AWS EC2 Setup
# Run on a fresh Ubuntu 22.04 EC2 instance
# Minimum: t3.medium (2 vCPU, 4 GB RAM)
# Recommended: t3.large (2 vCPU, 8 GB RAM)
# Storage: 100 GB+ gp3 EBS
# =============================================================
set -euo pipefail

echo "==> Updating system packages"
apt-get update && apt-get upgrade -y

echo "==> Installing system dependencies"
apt-get install -y \
    docker.io \
    docker-compose \
    git \
    python3.12 \
    python3.12-venv \
    python3-pip \
    postgresql-client \
    htop \
    tmux \
    curl \
    jq

echo "==> Enabling Docker"
systemctl enable docker
systemctl start docker
usermod -aG docker ubuntu

echo "==> Cloning project"
if [ ! -d /opt/bybit_bot ]; then
    git clone https://github.com/YOUR_REPO/bybit_futures_bot.git /opt/bybit_bot
fi
cd /opt/bybit_bot

echo "==> Creating Python virtual environment"
python3.12 -m venv /opt/bybit_bot/.venv
source /opt/bybit_bot/.venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Copying environment template"
if [ ! -f /opt/bybit_bot/.env ]; then
    cp .env.example .env
fi

echo "==> Installing systemd service"
cp deployment/bybit-bot.service /etc/systemd/system/bybit-bot.service
systemctl daemon-reload
systemctl enable bybit-bot

echo "============================================"
echo "  Setup complete."
echo ""
echo "  Next steps:"
echo "    1. Edit /opt/bybit_bot/.env with your keys"
echo "    2. Run: bash /opt/bybit_bot/deployment/timescaledb_setup.sh"
echo "    3. Run: systemctl start bybit-bot"
echo "============================================"
