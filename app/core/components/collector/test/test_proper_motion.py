# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import pytest
from astropy.time import Time
from lucupy.minimodel import SiderealTarget, TargetType

from app.core.components.collector import Collector


@pytest.fixture
def time():
    return Time('2022-03-31 12:00:00', scale='utc')


@pytest.fixture
def target():
    return SiderealTarget('', set(), TargetType.OTHER,
                          ra=78.85743749999999,
                          dec=15.630119444444444,
                          pm_ra=-5.718,
                          pm_dec=-5.781,
                          epoch=2000.0)


def test_proper_motion(target, time):
    """
    Test that the sun location is at the equator at J2000.
    """
    coord = Collector._calculate_proper_motion(target, time)
    assert abs(coord.ra.deg - 78.85740081) < 1e-5
    assert abs(coord.dec.deg - 15.63008372) < 1e-5
