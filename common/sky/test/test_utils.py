import pytest
from astropy.time import Time
from astropy.coordinates import EarthLocation, Angle, PrecessedGeocentricFrame, SkyCoord
import astropy.units as u
import numpy as np
from common.sky.utils import local_sidereal_time, current_geocent_frame, min_max_alt
from common.sky.utils import hour_angle_to_angle, true_airmass
from fixtures import midnight, location, coord

# All values are calculated using the following code:
# https://github.com/jrthorstensen/thorsky

@pytest.fixture
def times():
    # Time array
    dt = 5. * u.min
    nt = 6
    return Time("2020-07-01 10:00:00", format='iso', scale='utc') + dt * np.arange(nt)


@pytest.mark.usefixtures("midnight", "location")
def test_local_sidereal_time_with_single_value(midnight, location):
    assert local_sidereal_time(midnight, location) == Angle(17.71182168, unit=u.hourangle)


def test_local_sidereal_time_with_time_array(times, location):
    ...

@pytest.mark.usefixtures("midnight")
def test_current_geocent_frame(midnight):
    assert current_geocent_frame(midnight) == PrecessedGeocentricFrame(equinox='J2020.500', obstime='J2000.000')

@pytest.mark.usefixtures("coord", "location")
def test_min_max_alt(coord, location):
    minalt, maxalt = min_max_alt(location.lat, coord.dec)
    assert minalt.to_value(u.deg) == -38.96425410833332
    assert maxalt.to_value(u.deg) == 78.61185700277774

@pytest.mark.usefixtures("coord", "location")
def test_hour_angle_to_angle(coord, location):
    assert hour_angle_to_angle(coord[0].dec, location.lat, 30. * u.deg).deg == 66.23270582380651

@pytest.mark.parametrize("alt, expected", [(30, 1.9927834307900005),
                                           (60, 1.154177519115136),
                                           (80, 1.0153814647198127),
                                           (90, 1.0),
                                           (100, 1.0153814647198127),
                                           (-20, -1.0)
                                           ])
def test_true_airmass(alt, expected):
    assert true_airmass(alt) == expected
