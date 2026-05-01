"""
Run the SL-width sim against the live 6-symbol set with recent data.

This wraps scripts/sl_width_sim.py and overrides:
  - SYMBOLS: live trading set (SOL/AVAX/WIF/DOGE/BTC/XRP)
  - FROM/TO: 2024-01-01 -> today
  - START_EQUITY: $1000 (matches the live test account)
  - RISK_PCT: 0.0025 (matches LIVE_RISK_PCT)

The sim still tests the same SL_MULTS list and runs both Mode A
(fixed R:R 5.4) and Mode B (fixed TP 6.5x).

Drop this in /opt/bybit_bot/scripts/ and execute with:
    cd /opt/bybit_bot && .venv/bin/python scripts/_sim_overrides.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import scripts.sl_width_sim as sim

sim.SYMBOLS = [
    "SOLUSDT", "AVAXUSDT", "WIFUSDT",
    "DOGEUSDT", "BTCUSDT", "XRPUSDT",
]
sim.FROM = "2024-01-01"
sim.TO = "2026-04-29"
sim.START_EQUITY = 1000.0
sim.RISK_PCT = 0.0025

if __name__ == "__main__":
    sim.main()
