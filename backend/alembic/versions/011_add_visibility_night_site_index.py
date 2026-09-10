
"""Add (night_date, site_id) index on visibility_data

The Visibility tab reads visibility_data by night and site: the "what is visible
tonight" list, and the per-night check that answers whether every expected
observation is stored. Neither existing index serves that access path —
ix_visibility_target_date leads with target_id and ix_visibility_observation
with observation_id — so both queries would fall back to a sequential scan of a
table holding one JSONB-carrying row per observation per night of the semester.

Revision ID: 011
Revises: 010
Create Date: 2026-07-29

"""
from typing import Sequence, Union

from alembic import op

# Required variables
revision: str = "011"
down_revision: Union[str, None] = "010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_visibility_night_site",
        "visibility_data",
        ["night_date", "site_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_visibility_night_site", table_name="visibility_data")
