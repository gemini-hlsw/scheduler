"""Initial schema: sites, night_events, targets, target_night_data

Revision ID: 001
Revises: 
Create Date: 2025-02-03

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Sites table
    op.create_table(
        "sites",
        sa.Column("id", sa.SmallInteger(), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False, comment="Degrees"),
        sa.Column("longitude", sa.Float(), nullable=False, comment="Degrees"),
        sa.Column("elevation", sa.Float(), nullable=False, comment="Meters"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    # Night events table
    op.create_table(
        "night_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("site_id", sa.SmallInteger(), nullable=False),
        sa.Column("night_date", sa.Date(), nullable=False),
        sa.Column("night_duration_minutes", sa.Integer(), nullable=False),
        sa.Column("sunset", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sunrise", sa.DateTime(timezone=True), nullable=False),
        sa.Column("night_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("night_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("midnight", sa.DateTime(timezone=True), nullable=False),
        sa.Column("twilight_evening_12", sa.DateTime(timezone=True), nullable=False),
        sa.Column("twilight_morning_12", sa.DateTime(timezone=True), nullable=False),
        sa.Column("moonrise", sa.DateTime(timezone=True), nullable=True),
        sa.Column("moonset", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sun_alt", sa.LargeBinary(), nullable=False),
        sa.Column("sun_az", sa.LargeBinary(), nullable=False),
        sa.Column("sun_par_ang", sa.LargeBinary(), nullable=True),
        sa.Column("moon_alt", sa.LargeBinary(), nullable=False),
        sa.Column("moon_az", sa.LargeBinary(), nullable=False),
        sa.Column("moon_par_ang", sa.LargeBinary(), nullable=True),
        sa.Column("moon_dist", sa.Float(), nullable=False, comment="AU"),
        sa.Column("sun_moon_ang", sa.LargeBinary(), nullable=False),
        sa.Column("local_sidereal_times", sa.LargeBinary(), nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["site_id"], ["sites.id"]),
        sa.UniqueConstraint("site_id", "night_date", name="uq_night_events_site_night"),
    )
    op.create_index("ix_night_events_night_date", "night_events", ["night_date"])

    # Targets table
    op.create_table(
        "targets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("is_sidereal", sa.Boolean(), nullable=False, default=True),
        sa.Column("base_ra", sa.Float(), nullable=True, comment="Degrees"),
        sa.Column("base_dec", sa.Float(), nullable=True, comment="Degrees"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_targets_is_sidereal", "targets", ["is_sidereal"])

    # Target night data table
    op.create_table(
        "target_night_data",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("target_id", sa.Integer(), nullable=False),
        sa.Column("site_id", sa.SmallInteger(), nullable=False),
        sa.Column("night_date", sa.Date(), nullable=False),
        sa.Column("night_duration_minutes", sa.Integer(), nullable=False),
        sa.Column("ra", sa.LargeBinary(), nullable=False, comment="Degrees"),
        sa.Column("dec", sa.LargeBinary(), nullable=False, comment="Degrees"),
        sa.Column("alt", sa.LargeBinary(), nullable=False, comment="Radians"),
        sa.Column("az", sa.LargeBinary(), nullable=False, comment="Radians"),
        sa.Column("hourangle", sa.LargeBinary(), nullable=False, comment="Radians"),
        sa.Column("airmass", sa.LargeBinary(), nullable=False),
        sa.Column("par_ang", sa.LargeBinary(), nullable=True, comment="Radians"),
        sa.Column("target_updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "computed_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["target_id"], ["targets.id"]),
        sa.ForeignKeyConstraint(["site_id"], ["sites.id"]),
        sa.UniqueConstraint(
            "target_id",
            "site_id",
            "night_date",
            name="uq_target_night_data_target_site_night",
        ),
    )
    op.create_index("ix_target_night_data_night_date", "target_night_data", ["night_date"])
    op.create_index("ix_target_night_data_target_id", "target_night_data", ["target_id"])


def downgrade() -> None:
    op.drop_table("target_night_data")
    op.drop_table("targets")
    op.drop_table("night_events")
    op.drop_table("sites")
