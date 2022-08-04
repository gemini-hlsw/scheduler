import pytest
from hypothesis import given, strategies as st

from services.horizons import Coordinates, HorizonsAngle, horizons_session
from app.common.minimodel import Site, NonsiderealTarget, TargetTag, TargetType
from datetime import datetime
import numpy as np


MAX_VALUE = 270 * np.pi / 180
MIN_VALUE = 90 * np.pi / 180


@pytest.fixture
def target():
    return NonsiderealTarget('Jupiter', set(), type=TargetType.BASE,
                             tag=TargetTag.MAJOR_BODY, des='jupiter', ra=np.array([]), dec=np.array([]))


@pytest.fixture
def session_parameters():
    return Site.GS, datetime(2019, 2, 1), datetime(2019, 2, 1, 23, 59, 59), 300


@given(
    st.lists(
        st.floats(allow_infinity=False, allow_nan=False),
        min_size=4,
        max_size=4,
    )
)
def test_angular_distace_between_values(values):
    """
    Angular Distance must be in [0, 180°]
    """
    a, b, c, d = values
    assert Coordinates(a, b).angular_distance(Coordinates(c, d)) <= 180


@given(
    st.lists(
        st.floats(allow_infinity=False, allow_nan=False),
        min_size=2,
        max_size=2,
    )
)
def test_angular_distace_between_any_point_and_itself(values):
    """
    Angular Distance must be zero between any point and itself
    """
    a, b = values
    assert Coordinates(a, b).angular_distance(Coordinates(a, b)) == 0


@given(
    st.lists(
        st.floats(allow_infinity=False, allow_nan=False),
        min_size=4,
        max_size=4,
    )
)
def test_angular_distace_symmetry(values):
    """
    Angular Distance must be symmetric to within 1µas
    """
    a, b, c, d = values
    phi_2 = Coordinates(a, b).angular_distance(Coordinates(c, d))
    phi_1 = Coordinates(c, d).angular_distance(Coordinates(a, b))
    delta_phi = phi_2 - phi_1
    assert HorizonsAngle.to_signed_microarcseconds(delta_phi) <= 1


@given(
    st.lists(
        st.floats(allow_infinity=False, allow_nan=False, max_value=MAX_VALUE, min_value=MIN_VALUE),
        min_size=4,
        max_size=4,
    )
)
def test_interpolation_by_angular_distance_for_factor_zero(values):
    """
    Interpolate should result in angular distance of 0° from `a` for factor 0.0, within 1µsec (15µas)
    """
    a, b, c, d = values
    delta = Coordinates(a, b).angular_distance(Coordinates(a, b).interpolate(Coordinates(c, d), 0.0))
    assert abs(HorizonsAngle.to_signed_microarcseconds(delta)) <= 15


@given(
    st.lists(
        st.floats(allow_infinity=False, allow_nan=False, max_value=MAX_VALUE, min_value=MIN_VALUE),
        min_size=4,
        max_size=4,
    )
)
def test_interpolation_by_angular_distance_for_factor_one(values):
    """
    Interpolate should result in angular distance of 0° from `b` for factor 1.0, within 1µsec (15µas)
    """
    a, b, c, d = values
    delta = Coordinates(c, d).angular_distance(Coordinates(a, b).interpolate(Coordinates(c, d), 1.0))
    assert abs(HorizonsAngle.to_signed_microarcseconds(delta)) <= 15


@given(
    st.lists(
        st.floats(allow_infinity=False, allow_nan=False, max_value=MAX_VALUE, min_value=MIN_VALUE),
        min_size=4,
        max_size=4,
    )
)
def test_interpolation_by_fractional_angular_separation(values):
    """
    Interpolate should be consistent with fractional angular separation, to within 20 µas
    """
    a, b, c, d = values
    
    sep = Coordinates(a, b).angular_distance(Coordinates(c, d))
    deltas = []

    for f in np.arange(-1.0, 2.0, 0.1):
        step_sep = HorizonsAngle.to_degrees(Coordinates(a, b).angular_distance(Coordinates(a, b).
                                                                               interpolate(Coordinates(c, d), f)))
        frac_sep = HorizonsAngle.to_degrees(sep * abs(f))
        frac_sep2 = frac_sep if frac_sep <= 180 else 360 - frac_sep
        deltas.append(abs(step_sep - frac_sep2))
    assert all(d < 20 for d in deltas)


def test_horizons_client_query(target: NonsiderealTarget,
                               session_parameters: dict):
    """
    HorizonsClient.query should return a list of Coordinates
    """
    with horizons_session(*session_parameters) as session:
        eph = session.get_ephemerides(target)
        assert isinstance(eph.coordinates, list)
        assert isinstance(eph.coordinates[0], Coordinates)
        assert eph.coordinates[0].ra == 4.476586331426079
        assert eph.coordinates[0].dec == -0.3880237049946405
