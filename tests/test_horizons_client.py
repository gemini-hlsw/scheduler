from common.helpers.helpers import angular_distance
import pytest 
from hypothesis import given, strategies as st
from horizons import horizons_session, HorizonsClient
from helpers import helpers







@given(a=st.floats, b=st.floats)
def test_interpolation_by_angular_distance(a, b):
    # "interpolate should result in angular distance of 0° from `a` for factor 0.0, within 1µsec (15µas)"
    delta = helpers.angular_distance(a, HorizonsClient.interpolate(a,b, 0.0))
    assert abs(toMicroArcSeconds(delta)) < 15

    # "interpolate should result in angular distance of 0° from `b` for factor 1.0, within 1µsec (15µas)"
    delta = helpers.angular_distance(b, HorizonsClient.interpolate(a, b, 1.0))
    assert abs(toMicroArcSeconds(delta)) < 15

@given(a=st.floats, b=st.floats)
def test_interpolation_by_fractional_angular_separation(a, b):

    µas180 = ...
    µas360 = ... 

    sep = angular_distance(a, b)
    deltas = [ ]

    for f in range(-1,2):
        (f/10.0)
        step_sep = HorizonsClient.interpolate(a, b, f/10.0)
        frac_sep = (toMicroarcseconds * abs(f/10.0))
        frac_sep2 = frac_sep if frac_sep <= µas180 else µas360 - frac_sep
        deltas.append(abs(step_sep - frac_sep2))
    
    assert all(d > 20 for d in deltas)