
"""Store per-slot moon distance in meters (packed) instead of nightly mean AU

The sky-brightness moon term needs the earth-moon distance in meters
(lucupy normalizes by EQUAT_RAD, which is in meters). The previous nightly
mean stored in AU was passed to lucupy as an AU Quantity, which astropy keeps
as an un-simplified AU/m composite whose raw value corrupted the lunar term
and classified every moon-up slot as SBANY. This restores the pre-sight
behavior: the per-slot topocentric Distance from lucupy's accurate_location,
packed like the other night-event arrays.

Existing night_events rows only hold the scalar mean and cannot be expanded
to per-slot arrays, so they are deleted (they are a recomputable cache).
visibility_data rows are deleted too: any row for an SB-constrained
observation was computed with the corrupted sky brightness and must be
recomputed by the aggregator/fill script.

Revision ID: 010
Revises: 009
Create Date: 2026-07-14

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# Required variables
revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DELETE FROM visibility_data")
    op.execute("DELETE FROM night_events")
    op.drop_column("night_events", "moon_dist")
    op.add_column(
        "night_events",
        sa.Column(
            "moon_dist",
            sa.LargeBinary(),
            nullable=False,
            comment="Meters, shape (night_duration_minutes,)",
        ),
    )


def downgrade() -> None:
    op.execute("DELETE FROM visibility_data")
    op.execute("DELETE FROM night_events")
    op.drop_column("night_events", "moon_dist")
    op.add_column(
        "night_events",
        sa.Column("moon_dist", sa.Float(), nullable=False, comment="AU"),
    )
