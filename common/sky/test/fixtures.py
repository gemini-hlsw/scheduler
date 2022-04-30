import pytest
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u


@pytest.fixture
def midnight():
    return Time("2020-07-01 9:25:00", format='iso', scale='utc')


@pytest.fixture
def coord():
    coords = ["1:12:43.2 +31:12:43", "1 12 43.2 +31 12 43"]
    return SkyCoord(coords, unit=(u.hourangle, u.deg), frame='icrs')


@pytest.fixture
def location():
    return EarthLocation.of_site('gemini_north')


@pytest.fixture
def test_time():
    return Time('2020-07-01 10:00:00.000', format='iso', scale='utc')
