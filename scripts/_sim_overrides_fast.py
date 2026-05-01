"""Run the optimised SL-width sim against the live 6-symbol set."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import scripts.sl_width_sim_fast as sim

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
