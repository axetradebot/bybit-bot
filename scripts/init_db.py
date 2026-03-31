"""
Run once after docker-compose up to apply Alembic migrations
and create TimescaleDB hypertables.
Usage: python scripts/init_db.py
"""
import asyncio
import os
import subprocess
import sys
from pathlib import Path

import asyncpg
import structlog

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.config import settings  # noqa: E402

log = structlog.get_logger()


async def verify_timescaledb(conn: asyncpg.Connection) -> None:
    result = await conn.fetchval(
        "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
    )
    if not result:
        raise RuntimeError(
            "TimescaleDB extension not found. Check your PostgreSQL setup."
        )
    log.info("timescaledb_verified", version=result)


async def apply_hypertables(conn: asyncpg.Connection) -> None:
    sql_path = Path(__file__).resolve().parents[1] / "src" / "db" / "hypertables.sql"
    sql = sql_path.read_text()
    await conn.execute(sql)
    log.info("hypertables_applied")


async def main() -> None:
    log.info("running_alembic_migrations")
    subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        check=True,
        env={**os.environ, "PYTHONPATH": str(project_root)},
    )

    dsn = settings.async_db_url.replace("+asyncpg", "").replace(
        "postgresql://", "postgresql://"
    )
    conn = await asyncpg.connect(dsn)
    try:
        await verify_timescaledb(conn)
        await apply_hypertables(conn)
        log.info("phase1_complete", message="Database foundation ready")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
