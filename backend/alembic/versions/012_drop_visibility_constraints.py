
"""Drop visibility_data.constraints

Each Stage 2 row stored a copy of the observation's constraints. Nothing read
it back: Stage 2 takes constraints from the request. Observations with long
repeating timing windows carried ~66 KB of JSON on every night, which made up
most of the table and was fetched and decoded on every full-row read.

DROP COLUMN only updates the catalog, but it needs an ACCESS EXCLUSIVE lock.
lock_timeout makes it fail fast if an aggregator run holds the table, rather
than queueing every read of visibility_data behind it; rerun the upgrade then.
The space is reclaimed as rows are rewritten, or at once with VACUUM FULL.

Revision ID: 012
Revises: 011
Create Date: 2026-10-07

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# Required variables
revision: str = "012"
down_revision: Union[str, None] = "011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.drop_column("visibility_data", "constraints")


def downgrade() -> None:
    # Restores the column shape only; the dropped values are gone and every
    # row comes back as an empty object.
    op.add_column(
        "visibility_data",
        sa.Column("constraints", JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.alter_column("visibility_data", "constraints", server_default=None)
