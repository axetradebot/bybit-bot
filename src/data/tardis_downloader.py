"""
Core Tardis download pipeline.

Streams normalized historical data from a local tardis-machine instance
(HTTP replay-normalized endpoint), maps records to ORM dicts via
bybit_mapper, and batch-upserts into TimescaleDB.
"""

import asyncio
import json
import urllib.parse
from datetime import date, timedelta

import aiohttp
import structlog
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from src.data.bybit_mapper import (
    map_derivative_ticker_to_funding,
    map_liquidation,
    map_trade,
    map_trade_bar,
)
from src.db.base import async_session_factory
from src.db.models import (
    Candles1m,
    Candles5m,
    FundingHistory,
    Liquidations,
    RawTrades,
)

log = structlog.get_logger()

EXCLUDE_FROM_UPDATE = frozenset({"id"})

DATA_TYPE_CONFIG = {
    "trade_bar_5m": {
        "tardis_type": "trade_bar_5m",
        "model": Candles5m,
        "mapper": map_trade_bar,
        "table_name": "candles_5m",
    },
    "trade_bar_1m": {
        "tardis_type": "trade_bar_1m",
        "model": Candles1m,
        "mapper": map_trade_bar,
        "table_name": "candles_1m",
    },
    "funding_rate": {
        "tardis_type": "derivative_ticker",
        "model": FundingHistory,
        "mapper": None,
        "table_name": "funding_history",
    },
    "liquidations": {
        "tardis_type": "liquidation",
        "model": Liquidations,
        "mapper": map_liquidation,
        "table_name": "liquidations",
    },
    "trades": {
        "tardis_type": "trade",
        "model": RawTrades,
        "mapper": map_trade,
        "table_name": "raw_trades",
    },
}


def _monthly_chunks(from_date: str, to_date: str) -> list[tuple[str, str]]:
    start = date.fromisoformat(from_date)
    end = date.fromisoformat(to_date)
    chunks: list[tuple[str, str]] = []
    current = start
    while current < end:
        next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
        chunk_end = min(next_month, end)
        chunks.append((current.isoformat(), chunk_end.isoformat()))
        current = chunk_end
    return chunks


def _resolve_upsert_cols(model_class):
    table = model_class.__table__
    pk_cols = [col.name for col in table.primary_key.columns]
    update_cols = [
        col.name
        for col in table.columns
        if col.name not in pk_cols and col.name not in EXCLUDE_FROM_UPDATE
    ]
    return table, pk_cols, update_cols


class TardisDownloader:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        exchange: str = "bybit",
        batch_size: int = 5000,
    ):
        self.base_url = base_url.rstrip("/")
        self.exchange = exchange
        self.batch_size = batch_size

    async def download(
        self,
        symbol: str,
        data_type: str,
        from_date: str,
        to_date: str,
    ) -> dict:
        if data_type not in DATA_TYPE_CONFIG:
            raise ValueError(
                f"Unknown data type '{data_type}'. "
                f"Valid: {list(DATA_TYPE_CONFIG)}"
            )

        if data_type == "funding_rate":
            return await self._download_funding(symbol, from_date, to_date)
        return await self._download_standard(symbol, data_type, from_date, to_date)

    # ------------------------------------------------------------------
    # Standard download (trade_bar, liquidation, trade)
    # ------------------------------------------------------------------

    async def _download_standard(
        self,
        symbol: str,
        data_type: str,
        from_date: str,
        to_date: str,
    ) -> dict:
        cfg = DATA_TYPE_CONFIG[data_type]
        tardis_type = cfg["tardis_type"]
        model_class = cfg["model"]
        mapper_fn = cfg["mapper"]
        table_name = cfg["table_name"]

        table, pk_cols, update_cols = _resolve_upsert_cols(model_class)

        total_rows = 0
        min_ts = None
        max_ts = None
        batch: list[dict] = []

        for chunk_from, chunk_to in _monthly_chunks(from_date, to_date):
            log.info(
                "chunk_start",
                symbol=symbol,
                data_type=data_type,
                period=f"{chunk_from} -> {chunk_to}",
            )

            async for record in self._stream(
                symbol, tardis_type, chunk_from, chunk_to
            ):
                row = mapper_fn(record)
                if row is None:
                    continue

                ts = row["timestamp"]
                if min_ts is None or ts < min_ts:
                    min_ts = ts
                if max_ts is None or ts > max_ts:
                    max_ts = ts

                batch.append(row)
                if len(batch) >= self.batch_size:
                    await self._upsert_batch(table, batch, pk_cols, update_cols)
                    total_rows += len(batch)
                    batch = []

                    if total_rows % 10_000 < self.batch_size:
                        log.info(
                            "progress",
                            symbol=symbol,
                            data_type=data_type,
                            rows_inserted=total_rows,
                            ts_range=f"{min_ts} -> {max_ts}",
                        )

        if batch:
            await self._upsert_batch(table, batch, pk_cols, update_cols)
            total_rows += len(batch)

        checksum = {
            "symbol": symbol,
            "data_type": data_type,
            "table_name": table_name,
            "total_rows": total_rows,
            "min_timestamp": str(min_ts) if min_ts else None,
            "max_timestamp": str(max_ts) if max_ts else None,
        }
        log.info("download_complete", **checksum)
        return checksum

    # ------------------------------------------------------------------
    # Funding-rate download (stateful: derivative_ticker -> funding_history)
    # ------------------------------------------------------------------

    async def _download_funding(
        self,
        symbol: str,
        from_date: str,
        to_date: str,
    ) -> dict:
        cfg = DATA_TYPE_CONFIG["funding_rate"]
        model_class = cfg["model"]
        table_name = cfg["table_name"]

        table, pk_cols, update_cols = _resolve_upsert_cols(model_class)

        total_rows = 0
        min_ts = None
        max_ts = None
        batch: list[dict] = []

        last_funding_ts: str | None = None
        last_record: dict | None = None

        for chunk_from, chunk_to in _monthly_chunks(from_date, to_date):
            log.info(
                "chunk_start",
                symbol=symbol,
                data_type="funding_rate",
                period=f"{chunk_from} -> {chunk_to}",
            )

            async for record in self._stream(
                symbol, "derivative_ticker", chunk_from, chunk_to
            ):
                current_funding_ts = record.get("fundingTimestamp")
                if not current_funding_ts or record.get("fundingRate") is None:
                    continue

                if (
                    last_funding_ts is not None
                    and current_funding_ts != last_funding_ts
                ):
                    row = map_derivative_ticker_to_funding(last_record)
                    ts = row["timestamp"]
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
                    if max_ts is None or ts > max_ts:
                        max_ts = ts

                    batch.append(row)
                    if len(batch) >= self.batch_size:
                        await self._upsert_batch(
                            table, batch, pk_cols, update_cols
                        )
                        total_rows += len(batch)
                        batch = []

                        if total_rows % 10_000 < self.batch_size:
                            log.info(
                                "progress",
                                symbol=symbol,
                                data_type="funding_rate",
                                rows_inserted=total_rows,
                                ts_range=f"{min_ts} -> {max_ts}",
                            )

                last_funding_ts = current_funding_ts
                last_record = record

        if batch:
            await self._upsert_batch(table, batch, pk_cols, update_cols)
            total_rows += len(batch)

        checksum = {
            "symbol": symbol,
            "data_type": "funding_rate",
            "table_name": table_name,
            "total_rows": total_rows,
            "min_timestamp": str(min_ts) if min_ts else None,
            "max_timestamp": str(max_ts) if max_ts else None,
        }
        log.info("download_complete", **checksum)
        return checksum

    # ------------------------------------------------------------------
    # Streaming + retry
    # ------------------------------------------------------------------

    async def _stream(
        self,
        symbol: str,
        tardis_type: str,
        from_date: str,
        to_date: str,
        max_retries: int = 5,
    ):
        options = {
            "exchange": self.exchange,
            "from": from_date,
            "to": to_date,
            "symbols": [symbol],
            "dataTypes": [tardis_type],
        }
        encoded = urllib.parse.quote(json.dumps(options))
        url = f"{self.base_url}/replay-normalized?options={encoded}"

        for attempt in range(max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=0, sock_read=300)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url) as resp:
                        if resp.status == 401:
                            raise PermissionError(
                                "Tardis API key is invalid or missing. "
                                "Set TARDIS_API_KEY in .env to a valid key. "
                                "Without a key, only the 1st of each month "
                                "is accessible."
                            )

                        if resp.status == 429:
                            wait = min(2 ** (attempt + 1) * 5, 120)
                            log.warning(
                                "rate_limited",
                                wait=wait,
                                attempt=attempt + 1,
                            )
                            await asyncio.sleep(wait)
                            continue

                        resp.raise_for_status()

                        async for raw_line in resp.content:
                            line = raw_line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            msg_type = record.get("type", "")
                            if msg_type in ("disconnect", "error"):
                                log.warning("stream_event", event=record)
                                continue

                            yield record
                return

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == max_retries - 1:
                    raise
                wait = min(2 ** (attempt + 1) * 2, 60)
                log.warning(
                    "stream_retry",
                    attempt=attempt + 1,
                    wait=wait,
                    error=str(exc),
                )
                await asyncio.sleep(wait)

    # ------------------------------------------------------------------
    # DB upsert
    # ------------------------------------------------------------------

    async def _upsert_batch(
        self,
        table,
        rows: list[dict],
        pk_cols: list[str],
        update_cols: list[str],
    ) -> None:
        present = set(rows[0].keys())
        actual_update = [c for c in update_cols if c in present]

        stmt = pg_insert(table).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=pk_cols,
            set_={col: stmt.excluded[col] for col in actual_update},
        )

        async with async_session_factory() as session:
            await session.execute(stmt)
            await session.commit()

    # ------------------------------------------------------------------
    # Verification helper
    # ------------------------------------------------------------------

    async def verify_table(self, table_name: str, symbol: str) -> dict:
        async with async_session_factory() as session:
            result = await session.execute(
                text(
                    f"SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
                    f"FROM {table_name} WHERE symbol = :symbol"
                ),
                {"symbol": symbol},
            )
            row = result.fetchone()
            return {
                "rows": row[0],
                "min_ts": row[1],
                "max_ts": row[2],
            }
