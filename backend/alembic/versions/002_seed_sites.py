"""Seed sites table with fixed observatory data

Revision ID: 002
Revises: 001
Create Date: 2025-02-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



SITES = [
    {
        "id": 1,
        "name": "Gemini North",
        "latitude": 19.82380145,
        "longitude": -155.46904675,
        "elevation": 4213.0,
    },
    {
        "id": 2,
        "name": "Gemini South",
        "latitude": -30.24074167,
        "longitude": -70.73668333,
        "elevation": 2750.0,
    },
]


def upgrade() -> None:
    sites_table = sa.table(
        "sites",
        sa.column("id", sa.SmallInteger),
        sa.column("name", sa.String),
        sa.column("latitude", sa.Float),
        sa.column("longitude", sa.Float),
        sa.column("elevation", sa.Float),
    )
    op.bulk_insert(sites_table, SITES)


def downgrade() -> None:
    op.execute("DELETE FROM sites WHERE id IN (1, 2)")