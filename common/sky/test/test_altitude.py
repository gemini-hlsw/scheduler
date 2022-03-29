import pytest
from common.sky.altitude import Altitude
from astropy.coordinates import Angle,Longitude
import astropy.units as u
from fixtures import coord, location


@pytest.mark.usefixtures("coord", "location")
def test_altitude_above_zero(coord, location):
    """
    Test that altitude is 0 at the equator.
    """
    alt, az, parallac = Altitude.above(coord[0].dec, -3.0 * u.hourangle, location.lat)

    assert alt.deg == Angle(0.84002209, unit=u.rad).deg
    assert az == Longitude(1.1339171, unit=u.radian)
    assert parallac == Angle(-1.6527975, unit=u.radian)
