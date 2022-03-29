import pytest
from common.sky.moon import Moon
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from fixtures import location


@pytest.fixture
def moon():
    return Moon(Time("2020-07-01 9:25:00", format='iso', scale='utc'))

@pytest.fixture
def test_time():
    return Time('2020-07-01 10:00:00.000', format='iso', scale='utc')


@pytest.mark.usefixtures("location")
def test_moon_accurate_location(moon, location):
    """
    Test that the moon location is accurate.
    """
    loc, dist = moon.accurate_location(location)
    assert loc.ra.value == 228.468331589897
    assert loc.dec.value == -15.238277160913936
    assert dist.value == 365468377.09845906

    
@pytest.mark.usefixtures("location")
def test_moon_low_precision_location(moon, location):
    """
    Test that the moon location is accurate.
    """
    loc, dist = moon.low_precision_location(location)
    assert loc.ra.value == 228.41771177093597
    assert loc.dec.value == -15.297127679461509
    assert dist.value == 365764976.08619356

@pytest.mark.usefixtures("location")
def test_moon_time_by_altitude(moon, test_time, location):
    """
    Test that the moon location is accurate.
    """
    alt = 0.0 * u.deg
    assert moon.time_by_altitude(alt, test_time, location) == Time('2020-07-01 12:38:21.732')
