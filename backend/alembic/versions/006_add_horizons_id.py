"""Add horizons_id to targets

Revision ID: 006
Revises: 005
Create Date: 2025-02-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "targets",
        sa.Column("horizons_id", sa.String(100), nullable=True, comment="NAIF ID or name for Horizons lookup"),
    )


def downgrade() -> None:
    op.drop_column("targets", "horizons_id")
    