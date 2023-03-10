# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime

import numpy as np
import pytest
from hypothesis import given, strategies as st
from hypothesis.strategies import composite
from lucupy.minimodel import Site, NonsiderealTarget, TargetTag, TargetType

from scheduler.services.horizons import Coordinates, HorizonsAngle, horizons_session


@composite
def coordinates(draw):
    # RA is in [0, 2π) radians.
    ra = draw(st.floats(min_value=0, max_value=np.pi, exclude_max=True))

    # Dec is in [-π, π] radians.
    dec = draw(st.floats(min_value=-np.pi / 2, max_value=np.pi/2))
    return Coordinates(ra, dec)


@pytest.fixture
def target():
    return NonsiderealTarget('Jupiter', frozenset(), type=TargetType.BASE,
                             tag=TargetTag.MAJOR_BODY, des='jupiter', ra=np.array([]), dec=np.array([]))


@pytest.fixture
def session_parameters():
    return Site.GS, datetime(2019, 2, 1), datetime(2019, 2, 1, 23, 59, 59), 300


@given(c1=coordinates(), c2=coordinates())
def test_angular_distance_between_values(c1, c2):
    """
    Angular Distance must always be in [0, 180°], or since in radians, equivalently [0, π].
    """
    assert c1.angular_distance(c2) <= np.pi


@given(c=coordinates())
def test_angular_distance_between_any_point_and_itself(c):
    """
    Angular Distance must be zero between any point and itself
    """
    assert c.angular_distance(c) == 0


@given(c1=coordinates(), c2=coordinates())
def test_angular_distance_symmetry(c1, c2):
    """
    Angular Distance must be symmetric to within 1µas
    """
    phi_2 = c1.angular_distance(c2)
    phi_1 = c2.angular_distance(c1)
    delta_phi = phi_2 - phi_1
    assert HorizonsAngle.to_signed_microarcseconds(delta_phi) <= 1


@given(c1=coordinates(), c2=coordinates())
def test_interpolation_by_angular_distance_for_factor_zero(c1, c2):
    """
    Interpolate should result in angular distance of 0° from `a` for factor 0.0, within 1µsec (15µas)
    """
    delta = c1.angular_distance(c1.interpolate(c2, 0.0))
    assert abs(HorizonsAngle.to_signed_microarcseconds(delta)) <= 15


@given(c1=coordinates(), c2=coordinates())
def test_interpolation_by_angular_distance_for_factor_one(c1, c2):
    """
    Interpolate should result in angular distance of 0° from `b` for factor 1.0, within 1µsec (15µas)
    """
    delta = c1.angular_distance(c1.interpolate(c2, 1.0))
    assert abs(HorizonsAngle.to_signed_microarcseconds(delta)) <= 15


@given(c1=coordinates(), c2=coordinates())
@pytest.mark.skip(reason='Cannot expect this level of precision from degrees.')
def test_interpolation_by_fractional_angular_separation(c1, c2):
    """
    Interpolate should be consistent with fractional angular separation, to within 20 µas
    """
    sep = c1.angular_distance(c2)
    deltas = []

    for f in np.arange(-1.0, 2.0, 0.1):
        step_sep = HorizonsAngle.to_degrees(c1.angular_distance(c1.interpolate(c2, f)))
        frac_sep = HorizonsAngle.to_degrees(sep * abs(f))
        # TODO: CHECK THIS
        frac_sep2 = frac_sep if frac_sep <= 180 else 360 - frac_sep
        deltas.append(abs(step_sep - frac_sep2))
    max_delta = max(deltas)
    assert max_delta < 55


def test_horizons_client_query(target: NonsiderealTarget,
                               session_parameters: dict):
    """
    HorizonsClient.query should return a list of Coordinates
    """
    with horizons_session(*session_parameters) as client:
        eph = client.get_ephemerides(target)
        assert isinstance(eph.coordinates, list)
        assert isinstance(eph.coordinates[0], Coordinates)
        assert eph.coordinates[0].ra == 4.476586331426079
        assert eph.coordinates[0].dec == -0.3880237049946405
