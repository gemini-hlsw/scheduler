import pytest
from astropy.time import Time
from common.sky.events import night_events
from fixtures import midnight, location
import pytz
import numpy.testing as nptest

@pytest.mark.usefixtures("midnight", "location")
def test_night_events(midnight, location):
    """
    Test that the night events are calculated correctly.
    """
    hst = pytz.timezone('Pacific/Honolulu')
    _, sunset, sunrise, even_12twi, morn_12twi, moonrise, moonset = night_events(midnight, location, hst)

    nptest.assert_almost_equal(sunset.jd, Time('2020-07-01 05:15:04.172').jd, decimal=2)
    nptest.assert_almost_equal(sunrise.jd, Time('2020-07-01 15:36:40.341').jd, decimal=2)
    nptest.assert_almost_equal(even_12twi.jd, Time('2020-07-01 05:58:33.537').jd, decimal=2)
    nptest.assert_almost_equal(morn_12twi.jd, Time('2020-07-01 14:53:11.218').jd, decimal=2)
    nptest.assert_almost_equal(moonrise.jd, Time('2020-07-01 00:50:37.391').jd, decimal=2)
    nptest.assert_almost_equal(moonset[1].jd, Time('2020-07-01 12:51:54.812').jd, decimal=2)
 