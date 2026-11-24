"""Add generated_plans.night_date

The night a plan covers, as the UI labels it: the UTC date of its morning twilight.
Lets the plan history list one night's plans without reading every plan's JSON.
Rows stored before this revision get it from their plan.

Revision ID: 014
Revises: 013
Create Date: 2026-11-24

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# Required variables
revision: str = "014"
down_revision: Union[str, None] = "013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "generated_plans",
        sa.Column("night_date", sa.Date(), nullable=True, comment="UTC date of the night's morning twilight"),
    )
    op.execute(
        """
        UPDATE generated_plans
        SET night_date = (
            (plan #>> '{nightPlans,nightTimeline,0,timeEntriesBySite,0,mornTwilight}')::timestamptz
            AT TIME ZONE 'UTC'
        )::date
        WHERE night_date IS NULL
        """
    )
    op.create_index("ix_generated_plans_night_date_created_at", "generated_plans", ["night_date", "created_at"])


def downgrade() -> None:
    op.drop_index("ix_generated_plans_night_date_created_at", table_name="generated_plans")
    op.drop_column("generated_plans", "night_date")
