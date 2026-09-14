# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import time
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from dateutil.parser import parse as parsedt
from lucupy.minimodel import CloudCover, Conditions, ImageQuality
from lucupy.minimodel import SkyBackground as OcsSkyBackground, WaterVapor as OcsWaterVapor
from lucupy.timeutils import sex2dec

from gpp_client.generated.enums import Instrument, SkyBackground, TimingWindowInclusion
from gpp_client.generated.scheduler_observations_updates import (
    SchedulerObservationsUpdatesObscalcUpdateValue as ObscalcValue,
)
from gpp_client.rest.models import VisibilityChanges

from scheduler.clients.gpp import gpp
from scheduler.config import config
from scheduler.services import logger_factory
from scheduler.services.sight.calculator.calculator import Calculator
from scheduler.services.sight.calculator.constants import site_key_from_instrument
from scheduler.services.sight.calculator.models import (
    ElevationType as SightElevationType,
    ObservationConstraints,
    ObservationRequest,
    TargetCreate,
    TimingWindow as SightTimingWindow,
)
from scheduler.services.sight.database.connection import session_scope
from scheduler.services.visibility_aggregator.aggregator import resolve_target_names

__all__ = [
    "sight_visibility_enabled",
    "site_key_from_instrument",
    "build_target_create",
    "build_conditions",
    "build_constraints",
    "expand_event_timing_windows",
    "calculate_and_store_visibility",
    "get_visibility_changes",
    "refresh_visibility_if_changed",
]

_logger = logger_factory.create_logger(__name__)


def sight_visibility_enabled() -> bool:
    """Whether visibility goes through the Sight service.
    """
    return str(config.collector.visibility_strategy).strip().lower() != "local"

# GPP SkyBackground -> Sight ``target_sb`` fraction. Same values the program
# provider uses (``GppProgramProvider._constraint_to_value``).
_SB_TO_FRACTION = {
    SkyBackground.DARKEST: 0.2,
    SkyBackground.DARK: 0.5,
    SkyBackground.GRAY: 0.8,
    SkyBackground.BRIGHT: 1.0,
}


def _to_utc_datetime(value) -> Optional[datetime]:
    """Coerce a GPP datetime scalar (str or datetime) to a tz-aware UTC datetime."""
    if value is None:
        return None
    dt = value if isinstance(value, datetime) else parsedt(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_date(value) -> date:
    """Coerce a GPP date scalar (``YYYY-MM-DD`` str or date/datetime) to a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _parse_epoch(value) -> float:
    """Parse a GPP epoch scalar to a float year.
    """
    if value is None:
        return 2000.0
    try:
        return float(value)
    except (TypeError, ValueError):
        s = str(value).strip()
        if s[:1] in ("B", "J"):
            try:
                return float(s[1:])
            except ValueError:
                return 2000.0
        return 2000.0


def build_target_create(value: ObscalcValue) -> Optional[TargetCreate]:
    """Build a Sight ``TargetCreate`` from the event's first asterism target.

    Returns ``None`` (caller skips) when there is no sidereal base target —
    non-sidereal targets are skipped for now, mirroring the aggregator.
    """
    asterism = value.target_environment.asterism if value.target_environment else []
    if not asterism:
        return None
    base = asterism[0]
    if base.sidereal is None:
        # Non-sidereal (or no coords): skip for now, same as the aggregator.
        return None
    return TargetCreate(
        name=str(base.name),
        is_sidereal=True,
        base_ra=sex2dec(base.sidereal.ra.hms, to_degree=True),
        base_dec=sex2dec(base.sidereal.dec.dms, to_degree=False),
        epoch=_parse_epoch(base.sidereal.epoch),
    )


def build_constraints(value: ObscalcValue, range_end: datetime) -> ObservationConstraints:
    """Build Sight ``ObservationConstraints`` (the Stage-2 inputs) from the event."""
    cs = value.constraint_set
    elevation = cs.elevation_range

    if elevation.air_mass is not None:
        elevation_type = SightElevationType.AIRMASS
        elevation_min = float(elevation.air_mass.min)
        elevation_max = float(elevation.air_mass.max)
    elif elevation.hour_angle is not None:
        elevation_type = SightElevationType.HOUR_ANGLE
        elevation_min = float(elevation.hour_angle.min_hours)
        elevation_max = float(elevation.hour_angle.max_hours)
    else:
        # Same fallback as GppProgramProvider.parse_elevation.
        elevation_type = SightElevationType.AIRMASS
        elevation_min, elevation_max = 1.0, 2.0

    target_sb = _SB_TO_FRACTION.get(cs.sky_background, 1.0)

    return ObservationConstraints(
        target_sb=target_sb,
        elevation_type=elevation_type,
        elevation_min=elevation_min,
        elevation_max=elevation_max,
        timing_windows=expand_event_timing_windows(value.timing_windows, range_end),
        has_resources=True,
        can_schedule=True,
    )


# GPP constraint preset -> absolute value. Same table as
# ``GppProgramProvider._constraint_to_value``, kept local so the event path does
# not pull in the program provider.
_PRESET_TO_VALUE = {
    'ZERO': 0.0, 'POINT_ONE': 0.1, 'POINT_TWO': 0.2, 'POINT_THREE': 0.3,
    'POINT_FOUR': 0.4, 'POINT_FIVE': 0.5, 'POINT_SIX': 0.6, 'POINT_EIGHT': 0.8,
    'ONE_POINT_ZERO': 1.0, 'ONE_POINT_TWO': 1.2, 'ONE_POINT_FIVE': 1.5,
    'TWO_POINT_ZERO': 2.0, 'THREE_POINT_ZERO': 3.0,
    'DARKEST': 0.2, 'DARK': 0.5, 'GRAY': 0.8, 'BRIGHT': 1.0,
    'VERY_DRY': 0.2, 'DRY': 0.5, 'MEDIAN': 0.8, 'WET': 1.0,
}

# Legacy OCS percentile bins, as in ``GppProgramProvider.parse_conditions``.
_CC_BINS = (0.1, 0.3, 1.0, 3.0)
_CC_BIN_VALUES = (0.5, 0.7, 0.8, 1.0)
_IQ_BINS = (0.45, 0.75, 1.05, 1.5)  # for r; should be wavelength dependent
_IQ_BIN_VALUES = (0.2, 0.7, 0.85, 1.0)


def _preset_value(preset) -> float:
    """Absolute value of a GPP constraint preset.

    The presets are string enums (``TWO_POINT_ZERO``), and pydantic hands back
    either the member or the bare string depending on how the payload was
    parsed, so both are accepted.
    """
    return _PRESET_TO_VALUE[str(getattr(preset, "value", preset))]


def _to_bin(value: float, bins, bin_values) -> float:
    """First bin the value fits in, or the loosest one when it fits none."""
    for limit, bin_value in zip(bins, bin_values):
        if value <= limit:
            return bin_value
    return bin_values[-1]


def build_conditions(value: ObscalcValue) -> Conditions:
    """The observation's constraints as OCS percentile bins.
    """
    cs = value.constraint_set
    elevation = cs.elevation_range
    # Hour-angle observations have no airmass limit; 2.0 is the provider's stand-in.
    x_max = float(elevation.air_mass.max) if elevation.air_mass is not None else 2.0

    iq_value = _preset_value(cs.image_quality)
    cc_value = _preset_value(cs.cloud_extinction)

    return Conditions(
        cc=CloudCover(_to_bin(cc_value, _CC_BINS, _CC_BIN_VALUES)),
        iq=ImageQuality(_to_bin(iq_value * x_max ** -0.6, _IQ_BINS, _IQ_BIN_VALUES)),
        sb=OcsSkyBackground(_preset_value(cs.sky_background)),
        wv=OcsWaterVapor(_preset_value(cs.water_vapor)),
    )


def expand_event_timing_windows(windows, range_end: datetime) -> List[SightTimingWindow]:
    """Expand event timing windows into flat Sight ``TimingWindow`` pairs.
    """
    out: List[SightTimingWindow] = []
    for tw in (windows or []):
        if tw.inclusion != TimingWindowInclusion.INCLUDE:
            continue
        start = _to_utc_datetime(tw.start_utc)
        if start is None:
            continue

        end = tw.end
        if end is None:
            out.append(SightTimingWindow(start=start, end=range_end))
            continue

        # TimingWindowEndAt: fixed end timestamp.
        if getattr(end, "at_utc", None) is not None:
            at = _to_utc_datetime(end.at_utc)
            if at is not None and at > start:
                out.append(SightTimingWindow(start=start, end=at))
            continue

        # TimingWindowEndAfter: duration (+ optional repeat).
        duration = timedelta(seconds=float(end.after.seconds))
        repeat = end.repeat
        if repeat is None:
            out.append(SightTimingWindow(start=start, end=start + duration))
            continue

        period = timedelta(seconds=float(repeat.period.seconds))
        count = None if repeat.times is None else int(repeat.times) + 1
        idx = 0
        while True:
            window_start = start + (period * idx if idx > 0 else timedelta(0))
            if count is None and window_start > range_end:
                break
            out.append(SightTimingWindow(start=window_start, end=window_start + duration))
            idx += 1
            if count is not None and idx >= count:
                break
            if period <= timedelta(0):  # guard against a zero/negative period loop
                break
    return out


async def calculate_and_store_visibility(
    value: ObscalcValue,
    observation_id: str,
    site_key: str,
) -> dict:
    """Compute and store visibility for one incoming observation.

    Builds the Sight inputs from the event, ensures the target/Stage-1 exist,
    then stores Stage-2 only for nights missing it across the program's active
    window. ``session_scope`` commits on success.
    """
    payload = build_target_create(value)
    if payload is None:
        _logger.info(
            f"Observation {observation_id}: no sidereal base target; "
            f"skipping visibility calculation."
        )
        return {"stored": 0, "skipped": "no_sidereal_target"}

    start_date = _to_date(value.program.active.start)
    end_date = _to_date(value.program.active.end)
    range_end = datetime.combine(end_date, datetime.min.time(), tzinfo=timezone.utc)

    request = ObservationRequest(
        observation_id=observation_id,
        target_name=payload.name,
        site_id=site_key,
        constraints=build_constraints(value, range_end),
    )

    t0 = time.perf_counter()
    async with session_scope() as session:
        calc = Calculator(session)
        # Ensure the target row exists.
        target = await calc.target_repo.get_by_name(payload.name)
        target_existed = target is not None
        if target is None:
            target = await calc.target_repo.create(
                name=payload.name,
                is_sidereal=payload.is_sidereal,
                base_ra=payload.base_ra,
                base_dec=payload.base_dec,
                pm_ra=payload.pm_ra,
                pm_dec=payload.pm_dec,
                epoch=payload.epoch,
                horizons_id=payload.horizons_id,
                tag=payload.tag,
            )
        # computes night events + Stage-1 on demand for the needed site/nights.
        result = await calc.store_missing_visibility([request], start_date, end_date)
    elapsed = time.perf_counter() - t0

    _logger.info(
        f"Observation {observation_id} ({site_key}, target '{payload.name}', "
        f"target_existed={target_existed}): stored {result.get('stored', 0)} new "
        f"visibility rows ({result.get('already_present', 0)}/{result.get('nights', 0)} "
        f"nights already present) over {start_date}..{end_date} in {elapsed:.2f}s."
    )
    return {**result, "target_existed": target_existed, "elapsed_seconds": round(elapsed, 2)}


# How far back an event looks for ODB visibility changes.
# This value is heuristic and might need modifications in the future.
_CHANGES_WINDOW = timedelta(minutes=2)


async def get_visibility_changes() -> VisibilityChanges:
    """ODB entities whose visibility inputs changed around the event being handled.

    """
    return await gpp.client.scheduler.get_visibility_changes(
        datetime.now(timezone.utc) - _CHANGES_WINDOW
    )


async def refresh_visibility_if_changed(
    value: ObscalcValue,
    observation_id: str,
    site_key: str,
    changes: VisibilityChanges,
) -> Optional[dict]:
    """Recompute one observation's stored visibility when the ODB reports it stale.

    The same invalidation the aggregator applies (``_apply_odb_changes``), for a
    single observation: a changed target has its stored coordinates overwritten
    (bumping ``updated_at``, which makes Stage 1 recompute as stale), and Stage 2
    rows are deleted so the store pass below refills them with the fresh inputs.

    Returns None when nothing changed for this observation, so the caller can
    tell "already up to date" from "refreshed".
    """
    payload = build_target_create(value)
    if payload is None:
        # Non-sidereal or no base target: nothing stored to refresh.
        return None

    obs_changed = str(value.id) in changes.observation_ids
    target_changed = False
    if changes.target_ids:
        # The event carries target names, the endpoint reports internal ids, so
        # the changed ids have to be resolved to compare them.
        changed_names = set((await resolve_target_names(sorted(changes.target_ids))).values())
        target_changed = payload.name in changed_names

    if not (obs_changed or target_changed):
        return None

    async with session_scope() as session:
        calc = Calculator(session)
        if target_changed:
            db_target = await calc.target_repo.get_by_name(payload.name)
            if db_target is not None:
                await calc.target_repo.update_fields(
                    db_target,
                    base_ra=payload.base_ra,
                    base_dec=payload.base_dec,
                    pm_ra=payload.pm_ra,
                    pm_dec=payload.pm_dec,
                    epoch=payload.epoch,
                )
        deleted = await calc.visibility_repo.delete_by_observation(observation_id)

    _logger.info(
        f"Observation {observation_id}: visibility inputs changed "
        f"(observation={obs_changed}, target={target_changed}); invalidated "
        f"{deleted} stored night(s), recomputing."
    )
    result = await calculate_and_store_visibility(value, observation_id=observation_id, site_key=site_key)
    return {**result, "invalidated": deleted, "obs_changed": obs_changed, "target_changed": target_changed}
