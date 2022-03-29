import pytest
from common.sky.sun import Sun
import astropy.units as u
from astropy.time import Time
from fixtures import midnight


def test_time():
    return Time('2020-07-01 10:00:00.000', format='iso', scale='utc')


@pytest.mark.usefixtures("midnight")
def test_sun_location_at_midnight(midnight):
    """
    Test that the sun location is at the equator at J2000.
    """
    pos = Sun.at(midnight)
    assert pos.ra.value == 100.88591931021546
    assert pos.dec.value == 23.058699652854724

@pytest.mark.usefixtures("location")
def test_sun_time_by_altitude(test_time, location):
    """
    Test that the sun location is at the equator at J2000.
    """
    alt = 0.0 * u.deg
    assert Sun.time_by_altitude(alt, test_time - (5.0 * u.hr), location) == Time("2020-07-01 05:01:07.885")
