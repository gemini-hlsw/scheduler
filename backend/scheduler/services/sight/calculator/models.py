from datetime import date, datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class ElevationType(str, Enum):
    NONE = "none"
    HOUR_ANGLE = "hour_angle"
    AIRMASS = "airmass"


class TimingWindow(BaseModel):
    """Time window when observation can occur."""
    start: datetime
    end: datetime


class ObservationConstraints(BaseModel):
    """Constraints for observation visibility calculation."""
    
    target_sb: float = Field(
        default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Sky background constraint (0.2, 0.5, 0.8, 1.0)"
    )
    elevation_type: ElevationType = Field(default=ElevationType.AIRMASS)
    elevation_min: float = Field(default=1.0, description="Min elevation value")
    elevation_max: float = Field(default=2.05, description="Max elevation value")
    timing_windows: list[TimingWindow] = Field(default_factory=list)
    has_resources: bool = Field(default=True)
    can_schedule: bool = Field(default=True)


class ObservationRequest(BaseModel):
    """Single observation request."""
    observation_id: str
    target_name: str
    site_id: str
    constraints: ObservationConstraints = Field(default_factory=ObservationConstraints)


class VisibilityRequest(BaseModel):
    """Request for visibility calculations."""
    observations: list[ObservationRequest]
    night_date: date


class VisibilityResult(BaseModel):
    """Result for a single observation."""
    observation_id: str
    target_name: str
    site: str
    night_date: date
    remaining_minutes: int
    visible_ranges: list[list[int]]


class CalculationResponse(BaseModel):
    """Response containing visibility results."""
    results: list[VisibilityResult]
    night_date: date
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TargetCreate(BaseModel):
    """Request to create a target."""
    name: str
    is_sidereal: bool = True
    base_ra: float = Field(ge=0.0, lt=360.0, description="Right ascension in degrees")
    base_dec: float = Field(ge=-90.0, le=90.0, description="Declination in degrees")
    pm_ra: float | None = Field(default=None, description="Proper motion RA in mas/yr")
    pm_dec: float | None = Field(default=None, description="Proper motion Dec in mas/yr")
    epoch: float | None = Field(default=2000.0, description="Coordinate epoch")
    horizons_id: str | None = Field(default=None, description="Horizons ID for non-sidereal")
    tag: str | None = Field(default=None, description="Target type: majorbody, asteroid, comet")


class TargetResponse(BaseModel):
    """Response for a single target."""
    name: str
    is_sidereal: bool
    base_ra: float | None
    base_dec: float | None
    pm_ra: float | None
    pm_dec: float | None
    epoch: float | None
    horizons_id: str | None
    tag: str | None
    
    class Config:
        from_attributes = True


class BulkTargetCreateRequest(BaseModel):
    """Request to create multiple targets."""
    targets: list[TargetCreate]
    start_date: date
    end_date: date


class BulkTargetCreateResponse(BaseModel):
    """Response for bulk target creation."""
    created: int
    failed: int
    targets: list[TargetResponse]
    errors: list[str] = Field(default_factory=list)


class BulkVisibilityRequest(BaseModel):
    """Request for visibility across multiple nights."""
    observations: list[ObservationRequest]
    start_date: date
    end_date: date


class BulkVisibilityResponse(BaseModel):
    """Response for bulk visibility query."""
    results: dict[str, list[VisibilityResult]]  # date string -> results
    start_date: date
    end_date: date
    total_nights: int
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def mask_to_ranges(mask: list[bool]) -> list[list[int]]:
    """
    Convert boolean visibility mask to list of [start, end] ranges.
    
    Example:
        [True, True, True, False, False, True, True] -> [[0, 2], [5, 6]]
    """
    ranges = []
    start = None
    
    for i, visible in enumerate(mask):
        if visible and start is None:
            start = i
        elif not visible and start is not None:
            ranges.append([start, i - 1])
            start = None
    
    # Handle case where mask ends with True
    if start is not None:
        ranges.append([start, len(mask) - 1])
    
    return ranges


class PrecomputeRequest(BaseModel):
    """Request to pre-compute Stage 1 data for existing targets."""
    start_date: date
    end_date: date
    target_names: list[str] | None = Field(
        default=None, 
        description="Specific targets to compute. If None, computes all targets."
    )
    site_ids: list[str] | None = Field(
        default=None,
        description="Specific sites. If None, computes for all sites (GN, GS)."
    )


class PrecomputeResponse(BaseModel):
    """Response for pre-compute request."""
    targets: int
    sites: int
    nights: int
    total_computations: int
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StoreVisibilityRequest(BaseModel):
    """Request to calculate and store visibility."""
    observations: list[ObservationRequest]
    start_date: date
    end_date: date


class StoreVisibilityResponse(BaseModel):
    """Response for store visibility request."""
    stored: int
    nights: int
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PrecalculatedVisibilityRequest(BaseModel):
    """Request to retrieve pre-calculated visibility."""
    observation_id: str | None = None
    target_name: str | None = None
    start_date: date
    end_date: date


class PrecalculatedVisibilityResult(BaseModel):
    """Single pre-calculated visibility result."""
    observation_id: str
    target_name: str
    site: str
    night_date: date
    remaining_minutes: int
    visible_ranges: list[list[int]]


class PrecalculatedVisibilityResponse(BaseModel):
    """Response for pre-calculated visibility query."""
    results: list[PrecalculatedVisibilityResult]
    total: int
    start_date: date
    end_date: date


class BulkPrecalculatedRequest(BaseModel):
    """Request to retrieve pre-calculated visibility by observations."""
    observation_ids: list[str]
    start_date: date
    end_date: date


class VisibilityByDate(BaseModel):
    """Visibility for a single date."""
    night_date: date
    site: str
    remaining_minutes: int
    visible_ranges: list[list[int]]


class VisibilityByTarget(BaseModel):
    """Visibility grouped by target."""
    nights: dict[str, VisibilityByDate]


class VisibilityByObservation(BaseModel):
    """Visibility grouped by observation."""
    targets: dict[str, VisibilityByTarget]


class BulkPrecalculatedResponse(BaseModel):
    """Response for bulk pre-calculated visibility query."""
    observations: dict[str, VisibilityByObservation]
    start_date: date
    end_date: date

class CumulativeVisibilityRequest(BaseModel):
    """Request for cumulative remaining minutes across a date range."""
    observation_ids: list[str]
    start_date: date
    end_date: date


class CumulativeByTarget(BaseModel):
    """Cumulative remaining minutes for a target."""
    site: str
    cumulative_remaining_minutes: int
    nights_with_visibility: int


class CumulativeByObservation(BaseModel):
    """Cumulative visibility grouped by observation."""
    targets: dict[str, CumulativeByTarget]


class CumulativeVisibilityResponse(BaseModel):
    """Response for cumulative remaining visibility."""
    observations: dict[str, CumulativeByObservation]
    start_date: date
    end_date: date
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BulkStage1Request(BaseModel):
    """Request to retrieve Stage 1 data in bulk."""
    target_names: list[str]
    site_ids: list[str]  # ["GN", "GS"]
    start_date: date
    end_date: date


class Stage1DataByDate(BaseModel):
    """Stage 1 data for a single date."""
    night_date: date
    site: str
    night_duration_minutes: int
    ra: list[float]
    dec: list[float]
    alt: list[float]  # radians
    az: list[float]  # radians
    hourangle: list[float]  # radians
    airmass: list[float]
    par_ang: list[float] | None  # radians


class Stage1DataByTarget(BaseModel):
    """Stage 1 data grouped by target."""
    nights: dict[str, Stage1DataByDate]  # "GN_2025-02-15" -> data


class BulkStage1Response(BaseModel):
    """Response for bulk Stage 1 query."""
    targets: dict[str, Stage1DataByTarget]  # target_name -> data
    start_date: date
    end_date: date



#### GreedyMax target stage1 info

class VisibleObservationsRequest(BaseModel):
    """Request to filter observations by visibility."""
    observations: list[ObservationRequest]
    night_date: date


class VisibleObservationsResponse(BaseModel):
    """Response containing only visible observations."""
    visible_observations: list[VisibilityResult]
    night_date: date
    total_requested: int
    total_visible: int
    computed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BulkStage1GreedyMaxRequest(BaseModel):
    """Request to retrieve minimal Stage 1 data in bulk."""
    target_names: list[str]
    site_ids: list[str]  # ["GN", "GS"]
    start_date: date
    end_date: date


class Stage1GreedyMaxByDate(BaseModel):
    """Minimal Stage 1 data for a single date."""
    night_date: date
    site: str
    night_duration_minutes: int
    ra: list[float]  # degrees
    dec: list[float]  # degrees
    alt: list[float]  # radians
    az: list[float]  # radians
    airmass: list[float]
    hourangle: list[float]  # radians


class Stage1GreedyMaxByTarget(BaseModel):
    """Minimal Stage 1 data grouped by target."""
    nights: dict[str, Stage1GreedyMaxByDate]  # "GN_2025-02-15" -> data


class BulkStage1GreedyMaxResponse(BaseModel):
    """Response for bulk minimal Stage 1 query."""
    targets: dict[str, Stage1GreedyMaxByTarget]  # target_name -> data
    start_date: date
    end_date: date
