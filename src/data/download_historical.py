"""
CLI entry point for downloading historical Bybit data from Tardis.dev.

Requires a running tardis-machine instance (see docker-compose.yml).

Usage:
    python src/data/download_historical.py \
        --symbols BTCUSDT ETHUSDT SOLUSDT \
        --from 2021-01-01 \
        --to 2024-12-31 \
        --data-types trade_bar_5m trade_bar_1m funding_rate liquidations \
        --batch-size 5000
"""

import argparse
import asyncio
import sys
from datetime import date, timedelta
from pathlib import Path

import structlog

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import settings  # noqa: E402
from src.data.tardis_downloader import DATA_TYPE_CONFIG, TardisDownloader  # noqa: E402

log = structlog.get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical Bybit data from Tardis.dev",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=settings.symbols,
        help="Symbols to download (default: from .env SYMBOLS)",
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        default="2021-01-01",
        help="Start date inclusive (ISO format, default: 2021-01-01)",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="End date non-inclusive (ISO format, default: yesterday)",
    )
    parser.add_argument(
        "--data-types",
        nargs="+",
        default=["trade_bar_5m", "trade_bar_1m", "funding_rate", "liquidations"],
        help="Data types to download (default: trade_bar_5m trade_bar_1m funding_rate liquidations)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="DB batch upsert size (default: 5000)",
    )
    parser.add_argument(
        "--exchange",
        default="bybit",
        help="Tardis exchange ID (default: bybit)",
    )
    parser.add_argument(
        "--tardis-url",
        default="http://localhost:8000",
        help="Tardis Machine base URL (default: http://localhost:8000)",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    downloader = TardisDownloader(
        base_url=args.tardis_url,
        exchange=args.exchange,
        batch_size=args.batch_size,
    )

    checksums: list[dict] = []

    for symbol in args.symbols:
        for data_type in args.data_types:
            if data_type not in DATA_TYPE_CONFIG:
                log.error("unknown_data_type", data_type=data_type)
                continue

            log.info(
                "download_start",
                symbol=symbol,
                data_type=data_type,
                from_date=args.from_date,
                to_date=args.to_date,
            )

            try:
                checksum = await downloader.download(
                    symbol=symbol,
                    data_type=data_type,
                    from_date=args.from_date,
                    to_date=args.to_date,
                )
                checksums.append(checksum)
            except Exception:
                log.exception(
                    "download_failed",
                    symbol=symbol,
                    data_type=data_type,
                )

    # ---- Verification table ----
    header = (
        f"{'Symbol':<12}| {'Data Type':<18}| {'Rows':>12}"
        f"| {'From':<24}| {'To':<24}"
    )
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(header)
    print(sep)

    for cs in checksums:
        table_name = cs["table_name"]
        symbol = cs["symbol"]

        try:
            db_stats = await downloader.verify_table(table_name, symbol)
            rows_str = f"{db_stats['rows']:,}"
            min_ts = str(db_stats["min_ts"] or "")[:23]
            max_ts = str(db_stats["max_ts"] or "")[:23]
        except Exception:
            rows_str = f"{cs['total_rows']:,}"
            min_ts = str(cs.get("min_timestamp", ""))[:23]
            max_ts = str(cs.get("max_timestamp", ""))[:23]

        print(
            f"{symbol:<12}| {table_name:<18}| {rows_str:>12}"
            f"| {min_ts:<24}| {max_ts:<24}"
        )

    print(f"{'=' * len(header)}")


if __name__ == "__main__":
    asyncio.run(main())
