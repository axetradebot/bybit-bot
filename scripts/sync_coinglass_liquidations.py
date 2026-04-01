"""
Fetch CoinGlass aggregated liquidation history and upsert into Postgres.

Run after migrations:
    alembic upgrade head

Usage:
    python scripts/sync_coinglass_liquidations.py
    python scripts/sync_coinglass_liquidations.py --symbol BTCUSDT

Schedule: every 4h+ on Hobbyist (data is 4h bars); one call per symbol
stays under COINGLASS_MAX_RPM.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from sqlalchemy import create_engine  # noqa: E402

from src.config import settings  # noqa: E402
from src.data.coinglass_liquidation import sync_liquidation_bars_to_db  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync CoinGlass liquidation bars")
    parser.add_argument(
        "--symbol",
        default=None,
        help="Single symbol (default: all SYMBOLS from settings)",
    )
    args = parser.parse_args()

    if not settings.coinglass_api_key:
        print("COINGLASS_API_KEY is not set — add it to .env")
        sys.exit(1)

    engine = create_engine(settings.sync_db_url)
    n = sync_liquidation_bars_to_db(engine, symbol=args.symbol)
    print(f"Upserted {n} row(s) total.")
    engine.dispose()


if __name__ == "__main__":
    main()
