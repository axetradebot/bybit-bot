"""
Download historical futures data from Binance public data repository.

Source: https://data.binance.vision/
- Kline (OHLCV) data with volume
- AggTrades with isBuyerMaker flag → compute buy/sell volume
- Funding rate history

Data is downloaded as zipped CSVs, parsed, and upserted into TimescaleDB
in the same format as the Tardis pipeline.

Usage:
    python src/data/binance_downloader.py \\
        --symbols BTCUSDT ETHUSDT SOLUSDT \\
        --from 2021-01-01 --to 2025-12-31
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert as pg_insert

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import settings  # noqa: E402
from src.db.models import Candles5m, FundingHistory  # noqa: E402

log = structlog.get_logger()

BASE_URL = "https://data.binance.vision/data/futures/um"
BATCH_SIZE = 5000


def _monthly_range(from_date: str, to_date: str):
    start = date.fromisoformat(from_date)
    end = date.fromisoformat(to_date)
    current = start.replace(day=1)
    while current < end:
        yield current
        next_month = (current + timedelta(days=32)).replace(day=1)
        current = next_month


def _download_zip(url: str) -> bytes | None:
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "bybit-bot/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _upsert_batch(engine, table, rows: list[dict], pk_cols, update_cols):
    if not rows:
        return
    present = set(rows[0].keys())
    actual_update = [c for c in update_cols if c in present]
    stmt = pg_insert(table).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=pk_cols,
        set_={col: stmt.excluded[col] for col in actual_update},
    )
    with engine.begin() as conn:
        conn.execute(stmt)


def _resolve_cols(model_class):
    table = model_class.__table__
    pk = [c.name for c in table.primary_key.columns]
    upd = [c.name for c in table.columns if c.name not in pk]
    return table, pk, upd


# -------------------------------------------------------------------
# Kline download → candles_5m (OHLCV + buy/sell volume from taker data)
# -------------------------------------------------------------------

def download_klines(engine, symbol: str, from_date: str, to_date: str) -> int:
    """
    Download 5m klines from Binance futures public data.
    Binance kline CSV columns:
      0: open_time, 1: open, 2: high, 3: low, 4: close, 5: volume,
      6: close_time, 7: quote_volume, 8: count, 9: taker_buy_volume,
      10: taker_buy_quote_volume, 11: ignore
    """
    table, pk_cols, update_cols = _resolve_cols(Candles5m)
    total = 0

    for month_start in _monthly_range(from_date, to_date):
        year_month = month_start.strftime("%Y-%m")
        url = (
            f"{BASE_URL}/monthly/klines/{symbol}/5m/"
            f"{symbol}-5m-{year_month}.zip"
        )

        log.info("downloading_klines", symbol=symbol, month=year_month)
        data = _download_zip(url)
        if data is None:
            log.warning("kline_not_found", symbol=symbol, month=year_month)
            continue

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
                    batch: list[dict] = []

                    for row in reader:
                        if len(row) < 11 or not row[0].isdigit():
                            continue

                        open_time_ms = int(row[0])
                        ts = datetime.fromtimestamp(
                            open_time_ms / 1000, tz=timezone.utc
                        )

                        volume = Decimal(row[5])
                        taker_buy_vol = Decimal(row[9])
                        taker_sell_vol = volume - taker_buy_vol

                        record = {
                            "symbol": symbol,
                            "timestamp": ts,
                            "open": Decimal(row[1]),
                            "high": Decimal(row[2]),
                            "low": Decimal(row[3]),
                            "close": Decimal(row[4]),
                            "volume": volume,
                            "buy_volume": taker_buy_vol,
                            "sell_volume": taker_sell_vol,
                            "volume_delta": taker_buy_vol - taker_sell_vol,
                            "quote_volume": Decimal(row[7]),
                            "trade_count": int(row[8]),
                        }
                        batch.append(record)

                        if len(batch) >= BATCH_SIZE:
                            _upsert_batch(engine, table, batch, pk_cols, update_cols)
                            total += len(batch)
                            batch = []

                    if batch:
                        _upsert_batch(engine, table, batch, pk_cols, update_cols)
                        total += len(batch)

        except Exception as exc:
            log.error("kline_parse_error",
                      symbol=symbol, month=year_month, error=str(exc))

        log.info("klines_progress", symbol=symbol, month=year_month, total=total)

    log.info("klines_complete", symbol=symbol, total_rows=total)
    return total


# -------------------------------------------------------------------
# Funding rate download
# -------------------------------------------------------------------

def download_funding(engine, symbol: str, from_date: str, to_date: str) -> int:
    """
    Download funding rate history from Binance.
    Binance premium-index-klines are used as a proxy, but the simpler
    approach is to use the REST API for funding rate history.
    """
    import json
    import time
    import urllib.request

    table, pk_cols, update_cols = _resolve_cols(FundingHistory)
    total = 0
    batch: list[dict] = []

    start_ts = int(datetime.fromisoformat(from_date).replace(
        tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.fromisoformat(to_date).replace(
        tzinfo=timezone.utc).timestamp() * 1000)

    current = start_ts

    log.info("downloading_funding", symbol=symbol)

    while current < end_ts:
        url = (
            f"https://fapi.binance.com/fapi/v1/fundingRate"
            f"?symbol={symbol}&startTime={current}&limit=1000"
        )

        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "bybit-bot/1.0"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                records = json.loads(resp.read())
        except Exception as exc:
            log.error("funding_fetch_error", error=str(exc))
            time.sleep(2)
            continue

        if not records:
            break

        for rec in records:
            ts = datetime.fromtimestamp(
                rec["fundingTime"] / 1000, tz=timezone.utc
            )
            mark_str = rec.get("markPrice")
            try:
                mark_price = Decimal(mark_str) if mark_str else None
            except Exception:
                mark_price = None

            row = {
                "symbol": symbol,
                "timestamp": ts,
                "funding_rate": Decimal(rec["fundingRate"]),
                "mark_price": mark_price,
            }
            batch.append(row)

            if len(batch) >= BATCH_SIZE:
                _upsert_batch(engine, table, batch, pk_cols, update_cols)
                total += len(batch)
                batch = []

        current = records[-1]["fundingTime"] + 1
        time.sleep(0.2)

    if batch:
        _upsert_batch(engine, table, batch, pk_cols, update_cols)
        total += len(batch)

    log.info("funding_complete", symbol=symbol, total_rows=total)
    return total


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download historical futures data from Binance public repo"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=settings.symbols,
        help="Symbols to download (default: from .env)",
    )
    parser.add_argument(
        "--from", dest="from_date", default="2021-01-01",
        help="Start date (default: 2021-01-01)",
    )
    parser.add_argument(
        "--to", dest="to_date",
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="End date (default: yesterday)",
    )
    parser.add_argument(
        "--skip-funding", action="store_true",
        help="Skip funding rate download",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    engine = create_engine(settings.sync_db_url)

    print("=" * 60)
    print("Binance Historical Data Downloader")
    print(f"Symbols: {args.symbols}")
    print(f"Period:  {args.from_date} -> {args.to_date}")
    print("=" * 60)

    for symbol in args.symbols:
        print(f"\n--- {symbol} ---")

        print(f"\n[1/2] Downloading 5m klines...")
        kline_rows = download_klines(
            engine, symbol, args.from_date, args.to_date
        )
        print(f"  Klines: {kline_rows:,} rows")

        if not args.skip_funding:
            print(f"\n[2/2] Downloading funding rates...")
            funding_rows = download_funding(
                engine, symbol, args.from_date, args.to_date
            )
            print(f"  Funding: {funding_rows:,} rows")

    # Verification
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)
    with engine.connect() as conn:
        for symbol in args.symbols:
            row = conn.execute(text(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
                "FROM candles_5m WHERE symbol = :s"
            ), {"s": symbol}).first()
            print(f"  {symbol} candles_5m: {row[0]:,} rows "
                  f"({str(row[1])[:10]} → {str(row[2])[:10]})")

            row_f = conn.execute(text(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
                "FROM funding_history WHERE symbol = :s"
            ), {"s": symbol}).first()
            print(f"  {symbol} funding:    {row_f[0]:,} rows")

    print("\n" + "=" * 60)
    print("Done! Next step: compute indicators")
    print("  python src/indicators/compute_all.py --symbol BTCUSDT --from 2021-01-01")
    print("=" * 60)


if __name__ == "__main__":
    main()
