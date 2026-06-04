from datetime import datetime, date
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    UniqueConstraint,
    func,
    JSON
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from enum import Enum

class TargetTag(str, Enum):
    """Type of non-sidereal target."""
    MAJORBODY = "majorbody"
    ASTEROID = "asteroid"
    COMET = "comet"
    OTHER = "other"

class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class Site(Base):
    """
    Observatory site.
    """
    __tablename__ = "sites"

    id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=False, comment="Degrees")
    longitude: Mapped[float] = mapped_column(Float, nullable=False, comment="Degrees")
    elevation: Mapped[float] = mapped_column(Float, nullable=False, comment="Meters")

    # Relationships
    night_events: Mapped[list["NightEvent"]] = relationship(back_populates="site")
    target_night_data: Mapped[list["TargetNightData"]] = relationship(back_populates="site")

    def __repr__(self) -> str:
        return f"<Site {self.name}>"


class NightEvent(Base):
    """
    Night events and ephemeris data for a single night at a single site.
    
    Timestamps are stored as UTC datetime.
    Arrays (sun_alt, moon_alt, etc.) have shape (night_duration_minutes,).
    """
    __tablename__ = "night_events"
    __table_args__ = (
        UniqueConstraint("site_id", "night_date", name="uq_night_events_site_night"),
        Index("ix_night_events_night_date", "night_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    site_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("sites.id"), nullable=False)
    night_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Night boundaries
    night_duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    sunset: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sunrise: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    night_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    night_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    midnight: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Twilights
    twilight_evening_12: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    twilight_morning_12: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Moon events
    moonrise: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    moonset: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Sun arrays (shape: night_duration_minutes) - stored in radians
    sun_alt: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    sun_az: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    sun_par_ang: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

    # Moon arrays (shape: night_duration_minutes) - stored in radians
    moon_alt: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    moon_az: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    moon_par_ang: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)
    moon_ra: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, comment="Degrees")
    moon_dec: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, comment="Degrees")
    moon_dist: Mapped[float] = mapped_column(Float, nullable=False, comment="AU")


    # Sun-moon angular separation
    sun_moon_ang: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Local sidereal time array (radians)
    local_sidereal_times: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Metadata
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        nullable=False, 
        server_default=func.now()
    )

    # Relationships
    site: Mapped["Site"] = relationship(back_populates="night_events")

    def __repr__(self) -> str:
        return f"<NightEvent site={self.site_id} night={self.night_date}>"


class Target(Base):
    """
    Astronomical target.
    
    For sidereal targets, base_ra and base_dec are the fixed coordinates.
    For non-sidereal targets, coordinates are computed via Horizons and stored
    in TargetNightData as arrays.
    """
    __tablename__ = "targets"
    __table_args__ = (
        Index("ix_targets_is_sidereal", "is_sidereal"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_sidereal: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    base_ra: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Degrees")
    base_dec: Mapped[Optional[float]] = mapped_column(Float, nullable=True, comment="Degrees")
    pm_ra: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Proper motion RA, mas/yr"
    )
    pm_dec: Mapped[float | None] = mapped_column(
        Float, nullable=True, comment="Proper motion Dec, mas/yr"
    )
    
    # Epoch for coordinates
    epoch: Mapped[float | None] = mapped_column(
        Float, nullable=True, default=2000.0, comment="Coordinate epoch, e.g. 2000.0"
    )

    #Nonsidereal targets
    horizons_id: Mapped[Optional[str]] = mapped_column(
        String(100), 
        nullable=True,
        comment="NAIF ID or name for Horizons lookup"
    )
    tag: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Target type: majorbody, asteroid, comet, other"
    )

    # Timestamps for cache invalidation
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    night_data: Mapped[list["TargetNightData"]] = relationship(back_populates="target")

    def __repr__(self) -> str:
        return f"<Target {self.name} sidereal={self.is_sidereal}>"


class TargetNightData(Base):
    """
    Stage 1 calculation results: target position data for a single night at a single site.
    
    All angular arrays are stored in radians with shape (night_duration_minutes,).
    RA/Dec arrays are in degrees.
    
    For sidereal targets, ra/dec arrays are constant (but stored as arrays for uniform handling).
    For non-sidereal targets, ra/dec vary per minute (from Horizons).
    """
    __tablename__ = "target_night_data"
    __table_args__ = (
        UniqueConstraint(
            "target_id", "site_id", "night_date", 
            name="uq_target_night_data_target_site_night"
        ),
        Index("ix_target_night_data_night_date", "night_date"),
        Index("ix_target_night_data_target_id", "target_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_id: Mapped[int] = mapped_column(Integer, ForeignKey("targets.id"), nullable=False)
    site_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("sites.id"), nullable=False)
    night_date: Mapped[date] = mapped_column(Date, nullable=False)

    # Denormalized for convenience (must match night_events.night_duration_minutes)
    night_duration_minutes: Mapped[int] = mapped_column(Integer, nullable=False)

    # Coordinate arrays (degrees) - shape: (night_duration_minutes,)
    ra: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, comment="Degrees")
    dec: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, comment="Degrees")

    # Position arrays (radians) - shape: (night_duration_minutes,)
    alt: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, comment="Radians")
    az: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, comment="Radians")
    hourangle: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, comment="Radians")
    airmass: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    par_ang: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True, comment="Radians")

    # Cache invalidation: snapshot of target.updated_at when computed
    target_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    target: Mapped["Target"] = relationship(back_populates="night_data")
    site: Mapped["Site"] = relationship(back_populates="target_night_data")

    def is_stale(self, target: Target) -> bool:
        """Check if this cached data is outdated."""
        return self.target_updated_at < target.updated_at

    def __repr__(self) -> str:
        return f"<TargetNightData target={self.target_id} site={self.site_id} night={self.night_date}>"
    
class VisibilityData(Base):
    """
    Pre-calculated Stage 2 visibility data.
    """
    __tablename__ = "visibility_data"
    __table_args__ = (
        UniqueConstraint(
            "observation_id", "target_id", "site_id", "night_date",
            name="uq_visibility_observation_night"
        ),
        Index("ix_visibility_target_date", "target_id", "night_date"),
        Index("ix_visibility_observation", "observation_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    
    # Observation reference
    observation_id: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Target and site
    target_id: Mapped[int] = mapped_column(Integer, ForeignKey("targets.id"), nullable=False)
    site_id: Mapped[int] = mapped_column(SmallInteger, ForeignKey("sites.id"), nullable=False)
    night_date: Mapped[date] = mapped_column(Date, nullable=False)
    
    # Results
    remaining_minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    visible_ranges: Mapped[list] = mapped_column(JSONB, nullable=False)  # [[start, end], ...]
    
    # Constraints used
    constraints: Mapped[dict] = mapped_column(JSONB, nullable=False)
    
    # Metadata
    computed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Relationships
    target: Mapped["Target"] = relationship()
    site: Mapped["Site"] = relationship()

    def __repr__(self) -> str:
        return f"<VisibilityData obs={self.observation_id} night={self.night_date}>"


class SchedulerCoordination(Base):
    """
    Cross-process coordination state, shared via Postgres between the always-on
    operation dyno and the one-off visibility-aggregator cron dyno.

    One row per coordinated activity, keyed by ``name``:
      - ``visibility_aggregator`` — set active while the aggregator runs so the
        operation process blocks plan creation until it finishes.
      - ``night_execution`` — set active while the operation process is computing
        a plan, so the aggregator does not start work concurrently.

    ``heartbeat_at`` plus a staleness threshold lets a consumer treat a crashed
    holder's row as inactive instead of wedging the interlock forever.
    """
    __tablename__ = "scheduler_coordination"

    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    holder: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, comment="Dyno / process that owns the row"
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    detail: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return f"<SchedulerCoordination {self.name} active={self.active}>"
        