"""coinglass liquidation bars

Revision ID: b8e4f2a1c9d0
Revises: 151e4707ce5b
Create Date: 2026-04-01

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b8e4f2a1c9d0"
down_revision: Union[str, None] = "151e4707ce5b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "coinglass_liquidation_bars",
        sa.Column("symbol", sa.String(length=20), nullable=False),
        sa.Column("bucket_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("interval", sa.String(length=10), nullable=False),
        sa.Column(
            "long_liquidation_usd",
            sa.Numeric(precision=30, scale=4),
            nullable=False,
        ),
        sa.Column(
            "short_liquidation_usd",
            sa.Numeric(precision=30, scale=4),
            nullable=False,
        ),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("symbol", "bucket_time", "interval"),
    )


def downgrade() -> None:
    op.drop_table("coinglass_liquidation_bars")
