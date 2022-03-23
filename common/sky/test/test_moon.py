import pytest
from common.sky.moon import Moon
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

@pytest.fixture
def moon():
    return Moon(Time('J2000'))

@pytest.fixture
def location():
    return EarthLocation(lat=0, lon=0, height=0)
    # or return a known location like Kitt Peak
    # kitt_peak = EarthLocation.from_geodetic(lon=-111.6*u.deg,
    #                                            lat=31.963333333333342*u.deg,
    #                                            height=2120*u.m)

def test_moon_accurate_location(moon, location):
    """
    Test that the moon location is accurate.
    """
    assert moon.accurate_location(location) == SkyCoord(0, 0, unit='radian')

def test_moon_low_precision_location(moon, location):
    """
    Test that the moon location is accurate.
    """
    assert moon.low_precision_location(location) == SkyCoord(0, 0, unit='radian')

def test_moon_time_by_altitude(moon, location):
    """
    Test that the moon location is accurate.
    """
    alt = ...
    assert moon.time_by_altitude(alt, Time('J2000'), location) == Time('J2000')
