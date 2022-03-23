import pytest
from common.sky.sun import Sun
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

@pytest.fixture
def location():
    return EarthLocation(lat=0, lon=0, height=0)
    # or return a known location like Kitt Peak
    # kitt_peak = EarthLocation.from_geodetic(lon=-111.6*u.deg,
    #                                            lat=31.963333333333342*u.deg,
    #                                            height=2120*u.m)

def test_sun_location_at_j2000():
    """
    Test that the sun location is at the equator at J2000.
    """

    assert Sun.at(Time('J2000')) == SkyCoord(0, 0, unit='radian')

def test_sun_time_by_altitude(location):
    """
    Test that the sun location is at the equator at J2000.
    """
    alt = ...
    assert Sun.time_by_altitude(alt, Time('J2000'), location)) == Time('J2000')
