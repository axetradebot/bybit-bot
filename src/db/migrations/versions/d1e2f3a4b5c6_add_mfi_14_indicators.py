"""add mfi_14 to indicators tables

Revision ID: d1e2f3a4b5c6
Revises: b8e4f2a1c9d0
Create Date: 2026-04-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "d1e2f3a4b5c6"
down_revision: Union[str, None] = "b8e4f2a1c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "indicators_5m",
        sa.Column("mfi_14", sa.Numeric(precision=10, scale=4), nullable=True),
    )
    op.add_column(
        "indicators_15m",
        sa.Column("mfi_14", sa.Numeric(precision=10, scale=4), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("indicators_15m", "mfi_14")
    op.drop_column("indicators_5m", "mfi_14")
