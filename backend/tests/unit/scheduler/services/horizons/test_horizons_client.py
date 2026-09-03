# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime, timedelta
from typing import Final, List, Tuple

import numpy as np
import pytest
import hypothesis
from hypothesis import given, note
from hypothesis import strategies as st
from hypothesis.strategies import composite
from lucupy.minimodel import Site, NonsiderealTarget, TargetName, TargetTag, TargetType

from scheduler.services.horizons import Coordinates, HorizonsClient, horizons_session


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
        tag=TargetTag.MAJORBODY,
        des='jupiter')


@pytest.fixture(scope='module')
def session_parameters() -> Tuple[Site, datetime, datetime, float]:
    return Site.GS, datetime(2019, 2, 1), datetime(2019, 2, 1, 23, 59, 59), 1.0


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
@pytest.mark.skip('Can fail for points that are nearly antipodal and take the "long way" around the sphere.')
@given(c1=coordinates(), c2=coordinates())
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


@pytest.fixture(scope='module')
def horizons_client(session_parameters: Tuple[Site, datetime, datetime, float]) -> HorizonsClient:
    site, start, end, time_slot_length = session_parameters
    return HorizonsClient(site=site, start=start, end=end, time_slot_length=time_slot_length)


def _angular_distance_arcsec(c1: Coordinates, c2: Coordinates) -> float:
    return _to_degrees(c1.angular_distance(c2)) * 3600


@given(c1=coordinates(), c2=coordinates(), c3=coordinates())
def test_interpolate_coords_length_matches_step_count(horizons_client: HorizonsClient, c1, c2, c3):
    """
    The number of interpolated points in each inter-knot segment should be the
    number of timeslots that fit (via integer division) in that segment's duration.
    """
    time_list = [datetime(2019, 2, 1, 0, 0, 0),
                 datetime(2019, 2, 1, 0, 2, 0),
                 datetime(2019, 2, 1, 0, 5, 0)]
    coord_list = [c1, c2, c3]
    timeslot_length = timedelta(seconds=30)

    interpolated_times, interpolated_coords = horizons_client.interpolate_coords(
        time_list, coord_list, timeslot_length)

    # First segment is 2 minutes / 30s = 4 slots, second is 3 minutes / 30s = 6 slots.
    assert len(interpolated_times) == 10
    assert len(interpolated_coords) == 10


@given(c1=coordinates(), c2=coordinates(), c3=coordinates())
def test_interpolate_coords_time_grid(horizons_client: HorizonsClient, c1, c2, c3):
    """
    Interpolated times must start at the first entry of time_list, be evenly spaced by
    timeslot_length, and never reach (or pass) the final entry of time_list.
    """
    time_list = [datetime(2019, 2, 1, 0, 0, 0),
                 datetime(2019, 2, 1, 0, 2, 0),
                 datetime(2019, 2, 1, 0, 5, 0)]
    coord_list = [c1, c2, c3]
    timeslot_length = timedelta(seconds=30)

    interpolated_times, _ = horizons_client.interpolate_coords(time_list, coord_list, timeslot_length)

    assert interpolated_times[0] == time_list[0]
    assert all(t < time_list[-1] for t in interpolated_times)
    for earlier, later in zip(interpolated_times, interpolated_times[1:]):
        assert later - earlier == timeslot_length


@given(c1=coordinates(), c2=coordinates(), c3=coordinates())
def test_interpolate_coords_passes_through_knots(horizons_client: HorizonsClient, c1, c2, c3):
    """
    The cubic spline is fit through the original coordinates, so the interpolated value
    at a time coinciding with an original (non-final) knot must match that knot's
    coordinates, to well within 1 microarcsecond.
    """
    time_list = [datetime(2019, 2, 1, 0, 0, 0),
                 datetime(2019, 2, 1, 0, 2, 0),
                 datetime(2019, 2, 1, 0, 5, 0)]
    coord_list = [c1, c2, c3]
    timeslot_length = timedelta(seconds=30)

    _, interpolated_coords = horizons_client.interpolate_coords(
        time_list, coord_list, timeslot_length)

    # Index 0 coincides with time_list[0], index 4 (4 slots into the first 2-minute
    # segment) coincides with time_list[1].
    for index, knot_coord in [(0, c1), (4, c2)]:
        delta_arcsec = _angular_distance_arcsec(interpolated_coords[index], knot_coord)
        note(f'index={index}, knot={knot_coord}, interpolated={interpolated_coords[index]}, '
             f'delta_arcsec={delta_arcsec}')
        assert delta_arcsec < 1e-6


def test_interpolate_coords_uneven_gap_truncates_extra_slots(horizons_client: HorizonsClient):
    """
    When a segment's duration is not an exact multiple of timeslot_length, the leftover
    time (less than one timeslot) is dropped rather than producing a partial slot.
    """
    time_list = [datetime(2019, 2, 1, 0, 0, 0), datetime(2019, 2, 1, 0, 1, 10)]
    coord_list = [Coordinates(0.1, 0.1), Coordinates(0.2, 0.2)]
    timeslot_length = timedelta(seconds=30)

    interpolated_times, interpolated_coords = horizons_client.interpolate_coords(
        time_list, coord_list, timeslot_length)

    # 70 seconds / 30 seconds = 2 whole slots (the trailing 10 seconds is dropped).
    assert len(interpolated_times) == 2
    assert len(interpolated_coords) == 2
    assert interpolated_times == [time_list[0], time_list[0] + timeslot_length]


def test_horizons_client_query(target: NonsiderealTarget,
                               session_parameters: dict):
    """
    HorizonsClient.query should return a list of Coordinates
    """
    with horizons_session(*session_parameters) as client:
        eph = client.get_ephemerides(target)

        # Note: these are in radians.
        assert eph.coordinates[0].ra == -1.8065989757535077
        assert eph.coordinates[0].dec == -0.3880237049946405
