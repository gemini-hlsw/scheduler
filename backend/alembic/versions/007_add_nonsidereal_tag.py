"""Add tag to targets

Revision ID: 007
Revises: 006
Create Date: 2025-02-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "targets",
        sa.Column("tag", sa.String(50), nullable=True, comment="Target type: majorbody, asteroid, comet, other"),
    )


def downgrade() -> None:
    op.drop_column("targets", "tag")