"""Shims that adapt lucupy data shapes into what sight's pure compute expects."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import List, Optional

import astropy.units as u
from lucupy.minimodel.target import NonsiderealTarget, SiderealTarget
from lucupy.minimodel.timingwindow import TimingWindow

from scheduler.services.sight.calculator.models import (
    ElevationType as SightElevationType,
    ObservationConstraints as SightObservationConstraints,
    TargetCreate as SightTargetCreate,
    TimingWindow as SightTimingWindow,
)


# Sight's stage1 ``_HorizonsSiteAdapter._COORDINATE_CENTERS`` keys on the
# DB-style ``Site.name`` ("Gemini North" / "Gemini South"), so the shim's
# ``name`` must match.
_LUCUPY_SITE_TO_DB_NAME = {
    'GN': 'Gemini North',
    'GS': 'Gemini South',
}


def site_shim(lucupy_site) -> SimpleNamespace:
    """SimpleNamespace mirror of sight's ``database.Site`` for in-memory compute."""
    loc = lucupy_site.location
    return SimpleNamespace(
        name=_LUCUPY_SITE_TO_DB_NAME[lucupy_site.name],
        latitude=loc.lat.deg,
        longitude=loc.lon.deg,
        elevation=loc.height.to(u.m).value,
    )


def target_shim(target) -> Optional[SimpleNamespace]:
    """SimpleNamespace mirror of sight's ``database.Target`` for in-memory compute."""
    if target is None:
        return None
    if isinstance(target, SiderealTarget):
        return SimpleNamespace(
            name=str(target.name),
            is_sidereal=True,
            base_ra=float(target.ra),
            base_dec=float(target.dec),
            pm_ra=float(target.pm_ra) if target.pm_ra is not None else None,
            pm_dec=float(target.pm_dec) if target.pm_dec is not None else None,
            epoch=float(target.epoch) if target.epoch is not None else 2000.0,
            horizons_id=None,
            tag=None,
        )
    if isinstance(target, NonsiderealTarget):
        return SimpleNamespace(
            name=str(target.name),
            is_sidereal=False,
            base_ra=0.0,
            base_dec=0.0,
            pm_ra=None,
            pm_dec=None,
            epoch=2000.0,
            horizons_id=str(target.des) if target.des is not None else None,
            tag=target.tag.name.lower() if target.tag is not None else None,
        )
    return None


def target_create(target) -> Optional[SightTargetCreate]:
    """Build a sight ``TargetCreate`` payload from a lucupy Target.

    Single home for the lucupy -> sight target mapping, shared by the
    visibility-aggregator and ``scripts/fill_sight.py``. Returns None for an
    unrecognised/None target. Non-sidereal targets are mapped too (with
    placeholder base_ra/base_dec, since the schema requires them); callers that
    only want sidereal targets check ``.is_sidereal``.
    """
    if target is None:
        return None
    if isinstance(target, SiderealTarget):
        return SightTargetCreate(
            name=str(target.name),
            is_sidereal=True,
            base_ra=float(target.ra),
            base_dec=float(target.dec),
            pm_ra=float(target.pm_ra) if target.pm_ra is not None else None,
            pm_dec=float(target.pm_dec) if target.pm_dec is not None else None,
            epoch=float(target.epoch) if target.epoch is not None else 2000.0,
        )
    if isinstance(target, NonsiderealTarget):
        return SightTargetCreate(
            name=str(target.name),
            is_sidereal=False,
            # Sight requires base_ra/base_dec; non-sidereal targets resolve via
            # horizons_id, so we send neutral placeholders.
            base_ra=0.0,
            base_dec=0.0,
            horizons_id=str(target.des) if target.des is not None else None,
            tag=target.tag.name.lower() if target.tag is not None else None,
        )
    return None


def expand_timing_windows(windows, range_end: datetime) -> List[SightTimingWindow]:
    """Expand lucupy TimingWindow repeats into flat sight TimingWindow pairs.

    Used by both the realtime collector (Stage-2 constraint construction) and
    scripts/fill_sight.py (bulk store). Single home so behaviour stays in sync.
    """
    out: List[SightTimingWindow] = []
    for tw in (windows or []):
        if tw.start is None or tw.duration is None:
            continue
        tw_start = (
            tw.start.to_datetime(timezone.utc)
            if hasattr(tw.start, 'to_datetime')
            else tw.start
        )
        if tw_start.tzinfo is None:
            tw_start = tw_start.replace(tzinfo=timezone.utc)

        if tw.repeat == TimingWindow.NON_REPEATING:
            count, period = 1, None
        elif tw.repeat == TimingWindow.FOREVER_REPEATING:
            count, period = None, tw.period
        else:
            count, period = tw.repeat + 1, tw.period

        idx = 0
        while True:
            offset = (period * idx) if (period is not None and idx > 0) else timedelta(0)
            window_start = tw_start + offset
            if count is None and window_start > range_end:
                break
            out.append(SightTimingWindow(start=window_start, end=window_start + tw.duration))
            idx += 1
            if count is not None and idx >= count:
                break
            if period is None:
                break
    return out


def stage2_constraints(
    obs,
    has_resources: bool,
    can_schedule: bool,
    range_end: datetime,
) -> SightObservationConstraints:
    """Build sight Stage-2 constraints from a lucupy Observation."""
    constraints = getattr(obs, 'constraints', None)
    if constraints is None:
        return SightObservationConstraints(
            has_resources=has_resources,
            can_schedule=can_schedule,
        )
    cond = getattr(constraints, 'conditions', None)
    target_sb = float(cond.sb.value) if (cond is not None and cond.sb is not None) else 1.0
    return SightObservationConstraints(
        target_sb=target_sb,
        elevation_type=SightElevationType(constraints.elevation_type.name.lower()),
        elevation_min=float(constraints.elevation_min),
        elevation_max=float(constraints.elevation_max),
        timing_windows=expand_timing_windows(
            getattr(constraints, 'timing_windows', None), range_end,
        ),
        has_resources=has_resources,
        can_schedule=can_schedule,
    )
