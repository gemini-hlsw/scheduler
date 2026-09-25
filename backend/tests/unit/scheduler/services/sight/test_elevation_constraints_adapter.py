# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

"""Sight receives the elevation limits that were given, not the derived ones."""

from lucupy.minimodel import Constraints, ElevationLimits, ElevationType

from scheduler.services.sight._temporary.lucupy_adapters import elevation_constraints
from scheduler.services.sight.calculator.models import ElevationType as SightElevationType


def test_hour_angle_limits_are_given_to_sight():
    elevation = ElevationLimits(elevation_type=ElevationType.HOUR_ANGLE,
                                ha_min=-2.0, ha_max=4.0, airmass_min=1.0, airmass_max=1.8)
    assert elevation_constraints(elevation) == (SightElevationType.HOUR_ANGLE, -2.0, 4.0)


def test_airmass_limits_are_given_to_sight():
    elevation = ElevationLimits(elevation_type=ElevationType.AIRMASS,
                                ha_min=-3.0, ha_max=3.0, airmass_min=1.0, airmass_max=1.5)
    assert elevation_constraints(elevation) == (SightElevationType.AIRMASS, 1.0, 1.5)


def test_no_elevation_type_is_given_to_sight_with_default_airmass():
    elevation = ElevationLimits(elevation_type=ElevationType.NONE,
                                ha_min=None, ha_max=None,
                                airmass_min=Constraints.DEFAULT_AIRMASS_ELEVATION_MIN,
                                airmass_max=Constraints.DEFAULT_AIRMASS_ELEVATION_MAX)
    assert elevation_constraints(elevation) == (SightElevationType.NONE,
                                                Constraints.DEFAULT_AIRMASS_ELEVATION_MIN,
                                                Constraints.DEFAULT_AIRMASS_ELEVATION_MAX)
