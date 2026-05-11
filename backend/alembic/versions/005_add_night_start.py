"""Add night_start and night_end to night_events

Revision ID: 005
Revises: 004
Create Date: 2025-02-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "night_events",
        sa.Column("night_start", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "night_events",
        sa.Column("night_end", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("night_events", "night_end")
    op.drop_column("night_events", "night_start")
