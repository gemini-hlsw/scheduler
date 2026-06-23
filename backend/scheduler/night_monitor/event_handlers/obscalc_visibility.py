# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import time
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from dateutil.parser import parse as parsedt
from lucupy.timeutils import sex2dec

from gpp_client.generated.enums import Instrument, SkyBackground, TimingWindowInclusion
from gpp_client.generated.scheduler_observations_updates import (
    SchedulerObservationsUpdatesObscalcUpdateValue as ObscalcValue,
)

from scheduler.config import config
from scheduler.services import logger_factory
from scheduler.services.sight.calculator.calculator import Calculator
from scheduler.services.sight.calculator.models import (
    ElevationType as SightElevationType,
    ObservationConstraints,
    ObservationRequest,
    TargetCreate,
    TimingWindow as SightTimingWindow,
)
from scheduler.services.sight.database.connection import session_scope

__all__ = [
    "sight_visibility_enabled",
    "site_key_from_instrument",
    "build_target_create",
    "build_constraints",
    "expand_event_timing_windows",
    "calculate_and_store_visibility",
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

# GPP Instrument -> Sight site key, for instruments whose enum name does not end
# in NORTH/SOUTH (those are handled by the suffix check below). Mirrors and
# extends ``GppProgramProvider._site_for_inst``.
_INSTRUMENT_TO_SITE_KEY = {
    Instrument.FLAMINGOS2: "GS",
    Instrument.GHOST: "GS",
    Instrument.GPI: "GS",
    Instrument.GSAOI: "GS",
    Instrument.ZORRO: "GS",
    Instrument.SCORPIO: "GS",
    Instrument.GNIRS: "GN",
    Instrument.NIRI: "GN",
    Instrument.IGRINS2: "GN",
    Instrument.ALOPEKE: "GN",
    Instrument.MAROON_X: "GN",
}


def site_key_from_instrument(instrument) -> Optional[str]:
    """Derive the Sight site key (``"GN"`` / ``"GS"``) from an
    observation's instrument.
    """
    if instrument is None:
        return None
    name = getattr(instrument, "name", str(instrument)).upper()
    if name.endswith("NORTH"):
        return "GN"
    if name.endswith("SOUTH"):
        return "GS"
    return _INSTRUMENT_TO_SITE_KEY.get(instrument)


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
