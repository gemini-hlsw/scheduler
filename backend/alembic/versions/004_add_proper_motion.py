"""Add proper motion fields to targets

Revision ID: 004
Revises: 003
Create Date: 2025-02-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "targets",
        sa.Column("pm_ra", sa.Float(), nullable=True, comment="Proper motion RA, mas/yr"),
    )
    op.add_column(
        "targets",
        sa.Column("pm_dec", sa.Float(), nullable=True, comment="Proper motion Dec, mas/yr"),
    )
    op.add_column(
        "targets",
        sa.Column("epoch", sa.Float(), nullable=True, comment="Coordinate epoch, e.g. 2000.0"),
    )


def downgrade() -> None:
    op.drop_column("targets", "epoch")
    op.drop_column("targets", "pm_dec")
    op.drop_column("targets", "pm_ra")