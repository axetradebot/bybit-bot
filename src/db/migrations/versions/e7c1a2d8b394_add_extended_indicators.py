"""add extended indicator columns (adx/+di/-di/cmf/willr/roc/etc.)

Revision ID: e7c1a2d8b394
Revises: d1e2f3a4b5c6
Create Date: 2026-05-03

These columns are produced by ``compute_all.py`` (TA_COL_MAP and
DERIVED_FLOAT_COLS) but were never present in the indicators_5m /
indicators_15m schemas, which caused the live writer's circuit breaker
to enter a permanent ``recover -> fail -> pause`` loop on every bar
("Unconsumed column names: adx_14, ..."). Live trading was unaffected
(the bot uses in-memory indicator buffers and trades_log JSONB
snapshots both have the data) but indicators_{5,15}m have not received
fresh writes since 2026-04-02.

After this migration runs the writer will succeed on the next 5m
boundary and analytical queries against indicators_{5,15}m start
collecting fresh data again. The Apr 2 -> migration-time gap stays as
NULL for these specific columns; back-filling from cached parquet is
possible later if desired but not required (trades_log retains the
indicator snapshot for every actually-traded bar).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e7c1a2d8b394"
down_revision: Union[str, None] = "d1e2f3a4b5c6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (column_name, sa.Numeric(precision, scale))
NEW_COLUMNS = [
    # Trend strength / directional movement (0-100)
    ("adx_14", sa.Numeric(precision=10, scale=4)),
    ("plus_di", sa.Numeric(precision=10, scale=4)),
    ("minus_di", sa.Numeric(precision=10, scale=4)),
    # Momentum / flow
    ("cmf_20", sa.Numeric(precision=10, scale=4)),
    ("willr_14", sa.Numeric(precision=10, scale=4)),
    ("roc_10", sa.Numeric(precision=10, scale=4)),
    # Candle / volume structure
    ("candle_body_ratio", sa.Numeric(precision=10, scale=4)),
    ("volume_ratio", sa.Numeric(precision=10, scale=4)),
    ("vwma_20", sa.Numeric(precision=20, scale=8)),
    ("obv_slope", sa.Numeric(precision=30, scale=8)),
]

TABLES = ("indicators_5m", "indicators_15m")


def upgrade() -> None:
    for table in TABLES:
        for col_name, col_type in NEW_COLUMNS:
            op.add_column(
                table,
                sa.Column(col_name, col_type, nullable=True),
            )


def downgrade() -> None:
    for table in TABLES:
        for col_name, _ in reversed(NEW_COLUMNS):
            op.drop_column(table, col_name)
