# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

"""The GPP program provider stores the equivalent hour angle and airmass limits in the Constraints."""

import pytest
from lucupy.minimodel import (ElevationType, NonsiderealTarget, ObservationClass, SiderealTarget, Site, TargetName,
                              TargetTag, TargetType)

from scheduler.core.programprovider.gpp import GppProgramProvider
from scheduler.core.sources.sources import Sources

SIDEREAL = SiderealTarget(name=TargetName('sidereal'), magnitudes=frozenset(), type=TargetType.BASE,
                          ra=150.0, dec=-60.0, pm_ra=0.0, pm_dec=0.0, epoch=2000.0)
NONSIDEREAL = NonsiderealTarget(name=TargetName('Mars'), magnitudes=frozenset(), type=TargetType.BASE,
                                des='499', tag=TargetTag.MAJORBODY)


@pytest.fixture
def provider() -> GppProgramProvider:
    return GppProgramProvider(frozenset({ObservationClass.SCIENCE}), Sources())


def _data(elevation_range: dict) -> dict:
    return {
        'constraint_set': {
            'cloud_extinction': 'POINT_ONE',
            'image_quality': 'ONE_POINT_ZERO',
            'sky_background': 'DARK',
            'water_vapor': 'WET',
            'elevation_range': elevation_range,
        },
        'timing_windows': [],
    }


def test_hour_angle_constraints_sidereal_target(provider):
    constraints = provider.parse_constraints(_data({'hour_angle': {'min_hours': -2.0, 'max_hours': 4.0}}),
                                             Site.GS, SIDEREAL)
    elevation = constraints.elevation
    assert elevation.elevation_type == ElevationType.HOUR_ANGLE
    assert (elevation.ha_min, elevation.ha_max) == (-2.0, 4.0)
    assert elevation.airmass_min == 1.0
    assert 1.0 < elevation.airmass_max < 2.3


def test_airmass_constraints_sidereal_target(provider):
    constraints = provider.parse_constraints(_data({'air_mass': {'min': 1.0, 'max': 1.5}}), Site.GS, SIDEREAL)
    elevation = constraints.elevation
    assert elevation.elevation_type == ElevationType.AIRMASS
    assert (elevation.airmass_min, elevation.airmass_max) == (1.0, 1.5)
    assert elevation.ha_min == pytest.approx(-elevation.ha_max)
    assert elevation.ha_max > 0.0


@pytest.mark.parametrize('base', [NONSIDEREAL, None])
def test_derived_limits_unknown_without_dec(provider, base):
    constraints = provider.parse_constraints(_data({'hour_angle': {'min_hours': -2.0, 'max_hours': 4.0}}),
                                             Site.GS, base)
    elevation = constraints.elevation
    assert (elevation.ha_min, elevation.ha_max) == (-2.0, 4.0)
    assert (elevation.airmass_min, elevation.airmass_max) == (None, None)
