# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Selector.match_conditions: water vapor is a hard cutoff next to IQ and CC, with no penalty for better WV."""

from types import SimpleNamespace

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from lucupy.minimodel import CloudCover, Conditions, ImageQuality, SkyBackground, TooType, VariantSnapshot, WaterVapor

from scheduler.core.components.selector import Selector


def _obs(wv: float):
    conditions = Conditions(cc=CloudCover(1.0), iq=ImageQuality(1.0), sb=SkyBackground(1.0), wv=WaterVapor(wv))
    return SimpleNamespace(constraints=SimpleNamespace(conditions=conditions), too_type=TooType.STANDARD)


def _variant(wv: float, num_timeslots: int = 3):
    snapshot = VariantSnapshot(
        iq=ImageQuality(1.0),
        cc=CloudCover(1.0),
        wv=WaterVapor(wv),
        wind_dir=Angle(0, unit=u.deg),
        wind_spd=0 * (u.m / u.s),
    )
    return snapshot.make_variant(num_timeslots)


def _match(required_wv: float, actual_wv: float) -> np.ndarray:
    # The selector instance is only used for IQ in arcseconds, so it is not needed here.
    return Selector.match_conditions(None, _obs(required_wv), _variant(actual_wv), rising=True, night_idx=0)


def test_water_vapor_worse_than_required_blocks_the_observation():
    np.testing.assert_array_equal(_match(required_wv=0.5, actual_wv=0.8), np.zeros(3))


def test_water_vapor_as_required_matches():
    np.testing.assert_array_equal(_match(required_wv=0.5, actual_wv=0.5), np.ones(3))


def test_water_vapor_better_than_required_is_not_penalized():
    np.testing.assert_array_equal(_match(required_wv=1.0, actual_wv=0.2), np.ones(3))
