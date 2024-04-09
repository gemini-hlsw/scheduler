# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime
from typing import Final, Tuple

import numpy as np
import pytest
import hypothesis
from hypothesis import given, note
from hypothesis import strategies as st
from hypothesis.strategies import composite
from lucupy.minimodel import Site, NonsiderealTarget, TargetName, TargetTag, TargetType

from scheduler.services.horizons import Coordinates, horizons_session


_MICROARCSECS_PER_DEGREE: Final[float] = 60 * 60 * 1000 * 1000


def _to_signed_microarcseconds(angle: float) -> float:
    """
    Convert an angle in radians to a signed microarcsecond angle.
    """
    degrees = _to_degrees(angle)
    if degrees > 180:
        degrees -= 360
    return degrees * _MICROARCSECS_PER_DEGREE


def _to_degrees(angle: float) -> float:
    """
    Convert an angle in radians to a signed degree angle.
    """
    return angle * 180.0 / np.pi


def _to_microarcseconds(angle: float) -> float:
    """
    Convert an angle in radians to a signed microarcsecond angle.
    """
    return _to_degrees(angle) * _MICROARCSECS_PER_DEGREE


@composite
def coordinates(draw) -> Coordinates:
    # RA is in [0, 2pi) radians.
    ra = draw(st.floats(min_value=0, max_value=2 * np.pi, exclude_max=True))

    # Dec is in (-pi / 2, pi / 2) radians.
    dec = draw(st.floats(min_value=-np.pi/2, exclude_min=True,
                         max_value=np.pi/2, exclude_max=True))
    return Coordinates(ra, dec)


@pytest.fixture
def target() -> NonsiderealTarget:
    return NonsiderealTarget(
        name=TargetName('Jupiter'),
        magnitudes=frozenset(),
        type=TargetType.BASE,
        tag=TargetTag.MAJOR_BODY,
        des='jupiter',
        ra=np.array([]),
        dec=np.array([]))


@pytest.fixture
def session_parameters() -> Tuple[Site, datetime, datetime, int]:
    return Site.GS, datetime(2019, 2, 1), datetime(2019, 2, 1, 23, 59, 59), 300


@given(c1=coordinates(), c2=coordinates())
def test_angular_distance_between_values(c1, c2):
    """
    Angular distance must always be, in radians, in the interval [0, pi] (i.e. [0, 180] degrees).
    """
    assert c1.angular_distance(c2) <= np.pi


@given(c=coordinates())
def test_angular_distance_between_any_point_and_itself(c):
    """
    Angular distance must be zero between any point and itself.
    """
    assert c.angular_distance(c) == 0


@given(c1=coordinates(), c2=coordinates())
def test_angular_distance_symmetry(c1, c2):
    """
    Angular distance must be symmetric to within 1 mas.
    """
    phi_2 = c1.angular_distance(c2)
    phi_1 = c2.angular_distance(c1)
    delta_phi = abs(phi_2 - phi_1)
    assert _to_signed_microarcseconds(delta_phi) <= 1


@given(c1=coordinates(), c2=coordinates())
def test_interpolation_by_angular_distance_for_factor_zero(c1, c2):
    """
    Interpolate should result in angular distance of 0 degrees from c1 to c2 for factor 0.0,
    within 1 microsecond (15 mas).
    """
    delta = c1.angular_distance(c1.interpolate(c2, 0.0))
    assert abs(_to_signed_microarcseconds(delta)) <= 15


@given(c1=coordinates(), c2=coordinates())
def test_interpolation_by_angular_distance_for_factor_one(c1, c2):
    """
    Interpolate should result in angular distance of 0 degrees from c1 to c2 for factor 1.0,
    within 1 microsecond (15 mas).
    """
    delta = c2.angular_distance(c1.interpolate(c2, 1.0))
    assert abs(_to_signed_microarcseconds(delta)) <= 15


# This test fails in a very small number of cases. The original test case in Scala is marked as being flaky.
# This seems to happen in the boundary case, i.e. the points are antipodal, which makes the distance is close to pi.
# In this case, it is possible for Coordinates.angular_separation to take the "longer way" around.
# Example of failing value in the past:
# c1 = Coordinates(ra=0.0, dec=1.5707963263853362)
# c2 = Coordinates(ra=0.0, dec=-1.5707963263853362)
# which leads to:
# max_delta = 3.1415926535897922
@given(c1=coordinates(), c2=coordinates())
# @pytest.mark.skip('Can fail for points that are nearly antipodal and take the "long way" around the sphere.')
def test_interpolation_by_fractional_angular_separation(c1, c2):
    """
    Interpolate should be consistent with fractional angular separation.
    """
    hypothesis.settings(verbosity=hypothesis.Verbosity.verbose)
    threshold = 1e-3

    sep = c1.angular_distance(c2)
    deltas = []

    # Step above the end to get the full range of points from 0.0 to 1.0 by density.
    density = 0.01

    # slerp is spherical linear interpolation.
    slerp_lower = 0.0  # Use -1.0 for extended slerp.
    slerp_upper = 1.0  # Use  2.0 for extended slerp.
    for ratio in np.arange(slerp_lower, slerp_upper + density, density):
        # Calculate the expected angular separation based on the ratio.
        # The abs is unnecessary if ratio never negative.
        frac_sep = sep * abs(ratio)

        # Interpolate and take the angular distance.
        step_sep = c1.interpolate(c2, ratio).angular_distance(c1)

        # Adjust for boundary cases as best as possible.
        frac_sep2 = frac_sep if frac_sep <= np.pi else 2 * np.pi - frac_sep
        deltas.append(abs(step_sep - frac_sep2))

    max_delta = max(deltas)
    note(f'Interpolate - angular separation fail: {c1}, {c2}.')
    assert max_delta < threshold


def test_horizons_client_query(target: NonsiderealTarget,
                               session_parameters: dict):
    """
    HorizonsClient.query should return a list of Coordinates
    """
    with horizons_session(*session_parameters) as client:
        eph = client.get_ephemerides(target)
        assert eph.coordinates[0].ra == 4.476586331426079
        assert eph.coordinates[0].dec == -0.3880237049946405
