import pytest
from app.common.sky.altitude import Altitude
from astropy.coordinates import Angle, Longitude
import astropy.units as u
from fixtures import coord, location
import numpy.testing as nptest


@pytest.mark.usefixtures("coord", "location")
def test_altitude_above_zero(coord, location):
    """
    Test that altitude is 0 at the equator.
    """
    alt, az, parallac = Altitude.above(coord[0].dec, -3.0 * u.hourangle, location.lat)

    expected_alt = Angle(0.84002209, unit=u.rad)
    expected_az = Longitude(1.1339171, unit=u.radian)
    expected_par = Angle(-1.6527975, unit=u.radian)

    nptest.assert_almost_equal(alt.rad, expected_alt.rad, decimal=3)
    nptest.assert_almost_equal(az.rad, expected_az.rad, decimal=3)
    nptest.assert_almost_equal(parallac.rad, expected_par.rad, decimal=3)
