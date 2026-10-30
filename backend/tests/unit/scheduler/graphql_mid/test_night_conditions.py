# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import astropy.units as u
from astropy.coordinates import Angle
from lucupy.minimodel import CloudCover, ImageQuality, VariantSnapshot, WaterVapor

from scheduler.graphql_mid.types import SConditions


def test_night_conditions_carry_water_vapor_and_wind():
    variant = VariantSnapshot(
        iq=ImageQuality(0.7),
        cc=CloudCover(0.5),
        wv=WaterVapor(0.8),
        wind_dir=Angle(90.0, unit=u.deg),
        wind_spd=5.0 * (u.m / u.s),
    )
    conditions = SConditions.from_computed_conditions(variant)
    assert (conditions.iq, conditions.cc, conditions.wv) == ("IQ70", "CC50", "WV80")
    assert conditions.wind_speed == 5.0
    assert conditions.wind_direction == 90.0
