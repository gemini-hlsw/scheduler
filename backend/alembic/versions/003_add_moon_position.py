"""Add moon_ra and moon_dec to night_events

Revision ID: 003
Revises: 002
Create Date: 2025-02-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "night_events",
        sa.Column("moon_ra", sa.LargeBinary(), nullable=True, comment="Degrees"),
    )
    op.add_column(
        "night_events",
        sa.Column("moon_dec", sa.LargeBinary(), nullable=True, comment="Degrees"),
    )


def downgrade() -> None:
    op.drop_column("night_events", "moon_dec")
    op.drop_column("night_events", "moon_ra")
