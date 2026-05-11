
"""Add visibility_data table

Revision ID: 008
Revises: 007
Create Date: 2025-03-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# Required variables
revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "visibility_data",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("observation_id", sa.String(255), nullable=False),
        sa.Column("target_id", sa.Integer(), sa.ForeignKey("targets.id"), nullable=False),
        sa.Column("site_id", sa.SmallInteger(), sa.ForeignKey("sites.id"), nullable=False),
        sa.Column("night_date", sa.Date(), nullable=False),
        sa.Column("remaining_minutes", sa.Integer(), nullable=False),
        sa.Column("visible_ranges", JSONB(), nullable=False),
        sa.Column("constraints", JSONB(), nullable=False),
        sa.Column("computed_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("observation_id", "target_id", "site_id", "night_date", name="uq_visibility_observation_night"),
    )
    op.create_index("ix_visibility_target_date", "visibility_data", ["target_id", "night_date"])
    op.create_index("ix_visibility_observation", "visibility_data", ["observation_id"])


def downgrade() -> None:
    op.drop_index("ix_visibility_observation")
    op.drop_index("ix_visibility_target_date")
    op.drop_table("visibility_data")
    