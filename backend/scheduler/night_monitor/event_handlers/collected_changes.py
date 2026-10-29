# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from math import isclose
from typing import List, Optional, Tuple

from lucupy.minimodel import Constraints, ElevationType, NonsiderealTarget, Observation, SiderealTarget
from lucupy.timeutils import sex2dec

from gpp_client.generated.scheduler_observations_updates import (
    SchedulerObservationsUpdatesObscalcUpdateValue as ObscalcValue,
)

from scheduler.core.programprovider.gpp import GppProgramProvider
from scheduler.services import logger_factory

__all__ = ['changes_from_collected']

_logger = logger_factory.create_logger(__name__)

# The event is converted by the same provider the Collector reads the ODB with, so both sides land in
# the same units and bins. A local copy of the conversion drifts: build_conditions keeps the legacy
# OCS bins and an IQ percentile, while the Collector stores IQ in arcsec, so the two never match.
# Only parse methods that never touch the sources are called, hence no Sources.
_PROVIDER = GppProgramProvider(frozenset(), None)


def _name(preset) -> str:
    """GPP enums are str enums, and a payload can hold the member or the bare string."""
    return str(getattr(preset, 'value', preset))


def _conditions_changes(collected: Observation, value: ObscalcValue) -> List[str]:
    planned = collected.constraints.conditions
    cs = value.constraint_set
    # x_max only matters for an IQ percentile, which the provider does not compute today.
    x_max = collected.constraints.elevation.airmass_max or Constraints.DEFAULT_AIRMASS_ELEVATION_MAX
    required = _PROVIDER.parse_conditions(
        {
            'cloud_extinction': _name(cs.cloud_extinction),
            'image_quality': _name(cs.image_quality),
            'sky_background': _name(cs.sky_background),
            'water_vapor': _name(cs.water_vapor),
        },
        x_max,
        None,
    )
    return [
        f'{getattr(planned, field)} -> {getattr(required, field)}'
        for field in ('cc', 'iq', 'sb', 'wv')
        if getattr(planned, field) != getattr(required, field)
    ]


def _elevation_changes(collected: Observation, value: ObscalcValue) -> List[str]:
    elevation = value.constraint_set.elevation_range
    # The provider tells the type apart by which key is present, so absent ranges are left out.
    data = {}
    if elevation.air_mass is not None:
        data['air_mass'] = {'min': elevation.air_mass.min, 'max': elevation.air_mass.max}
    if elevation.hour_angle is not None:
        data['hour_angle'] = {
            'min_hours': elevation.hour_angle.min_hours,
            'max_hours': elevation.hour_angle.max_hours,
        }
    new_min, new_max, new_type = _PROVIDER.parse_elevation(data)

    stored = collected.constraints.elevation
    if stored.elevation_type is ElevationType.HOUR_ANGLE:
        old_min, old_max = stored.ha_min, stored.ha_max
    else:
        old_min, old_max = stored.airmass_min, stored.airmass_max

    if (
        stored.elevation_type is new_type
        and isclose(float(old_min), float(new_min), abs_tol=1e-6)
        and isclose(float(old_max), float(new_max), abs_tol=1e-6)
    ):
        return []
    return [
        f'elevation {stored.elevation_type.name} {float(old_min):g}-{float(old_max):g} -> '
        f'{new_type.name} {float(new_min):g}-{float(new_max):g}'
    ]


def _timing_window_data(tw) -> dict:
    """An event timing window in the shape the provider parses."""
    end = tw.end
    if end is None:
        end_data = None
    elif getattr(end, 'at_utc', None) is not None:
        end_data = {'at_utc': str(end.at_utc)}
    else:
        repeat = end.repeat
        end_data = {
            'after': {'seconds': float(end.after.seconds)},
            'repeat': None
            if repeat is None
            else {'period': {'seconds': float(repeat.period.seconds)}, 'times': repeat.times},
        }
    return {'inclusion': _name(tw.inclusion), 'start_utc': str(tw.start_utc), 'end': end_data}


def _timing_window_changes(collected: Observation, value: ObscalcValue) -> List[str]:
    def key(tw):
        return tw.start.isot, tw.duration, tw.repeat, tw.period

    old = [key(tw) for tw in collected.constraints.timing_windows]
    new = [
        key(_PROVIDER.parse_timing_window(_timing_window_data(tw)))
        for tw in value.scheduling_constraints.timing_windows
    ]
    return [] if old == new else [f'timing windows changed ({len(old)} -> {len(new)})']


def _target_key(target) -> Optional[Tuple]:
    if isinstance(target, SiderealTarget):
        return str(target.name), round(target.ra, 6), round(target.dec, 6)
    if isinstance(target, NonsiderealTarget):
        return str(target.name), str(target.des)
    return None


def _target_changes(collected: Observation, value: ObscalcValue) -> List[str]:
    # Proper motion and epoch are not in the event, so only the name and position are compared.
    stored = collected.targets[0] if collected.targets else None
    asterism = value.target_environment.asterism
    if stored is None or stored.name == GppProgramProvider._EMPTY_BASE_TARGET.name or not asterism:
        # Nothing to compare: the Collector had no base target, e.g. it found no sidereal or
        # nonsidereal block under an explicit base and fell back to the empty one.
        return []

    base = asterism[0]
    if base.sidereal is not None:
        new = (
            str(base.name),
            round(sex2dec(base.sidereal.ra.hms, to_degree=True), 6),
            round(sex2dec(base.sidereal.dec.dms, to_degree=False), 6),
        )
    elif base.nonsidereal is not None:
        new = str(base.name), str(base.nonsidereal.des)
    else:
        return []

    old = _target_key(stored)
    if old == new:
        return []
    if old is None or old[0] != new[0]:
        return [f'base target {stored.name} -> {new[0]}']
    return [f'base target {new[0]} moved']


def changes_from_collected(collected: Observation, value: ObscalcValue) -> List[str]:
    """
    What the event changed, compared with the observation the Collector read for the plan in effect.

    The obscalc update carries the whole observation as the ODB has it now, not a diff, so each field
    the Scheduler reads is compared in turn. Only what the event carries can be compared: an edit to
    the sequence (exposure times, steps, the execution digest) reads as no change here.

    Args:
        collected (Observation): the observation as the Collector stored it.
        value (ObscalcValue): the observation as the ODB reports it now.

    Returns:
        List[str]: one readable entry per change, empty when nothing comparable moved.
    """
    found = []

    state = _name(value.workflow.value.state)
    if collected.status.name != state:
        found.append(f'status {collected.status.name} -> {state}')

    if collected.constraints is None:
        # The Collector keeps no constraints without a wavelength, so there is nothing to diff against.
        return found

    for section in (_conditions_changes, _elevation_changes, _timing_window_changes, _target_changes):
        try:
            found.extend(section(collected, value))
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            # One unreadable part of the payload must not hide what the others say.
            _logger.warning(f'Could not compare {section.__name__[1:]} of {collected.id.id}: {exc}')
    return found
