"""Add generated_plans table

One row per plan the real-time engine generates, with the event that triggered it.
The plan is JSONB because its format still changes often.

Revision ID: 013
Revises: 012
Create Date: 2026-11-23

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# Required variables
revision: str = "013"
down_revision: Union[str, None] = "012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "generated_plans",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            comment="When the plan was stored",
        ),
        sa.Column(
            "app_version",
            sa.String(100),
            nullable=False,
            comment="APP_VERSION of the server that made it, e.g. 1.4.0-staging-realtime",
        ),
        sa.Column(
            "mode",
            sa.String(20),
            nullable=False,
            comment="SchedulerModes the server runs in: operation, simulation",
        ),
        sa.Column("event_type", sa.String(100), nullable=False, comment="Event class, e.g. OnDemandScheduleEvent"),
        sa.Column("event_description", sa.Text(), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("event_site", sa.String(10), nullable=True, comment="Null for both sites"),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            comment="in_effect, proposed, accepted, discarded, superseded, expired",
        ),
        sa.Column("proposal_id", sa.String(64), nullable=True),
        sa.Column("decided_by", sa.String(255), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("plan", JSONB(), nullable=False, comment="{nightPlans, plansSummary} as sent to the UI"),
    )
    op.create_index("ix_generated_plans_created_at", "generated_plans", ["created_at"])
    op.create_index("ix_generated_plans_mode_created_at", "generated_plans", ["mode", "created_at"])
    op.create_index("ix_generated_plans_proposal_id", "generated_plans", ["proposal_id"], unique=True)


def downgrade() -> None:
    # Dropping the table drops its indexes too. Naming them here would fail on a database
    # migrated with an earlier draft of this revision that lacked one of them.
    op.drop_table("generated_plans")
