import pytest
from common.sky.altitude import Altitude
from astropy.coordinates import Angle

@pytest.fixture
def dec():
    return Angle(0, unit='deg')

@pytest.fixture
def ha():
    return Angle(0, unit='hourangle')

@pytest.fixture
def lat():
    return Angle(0, unit='deg')

def test_altitude_above_zero(dec, ha, lat):
    """
    Test that altitude is 0 at the equator.
    """
    alt, az, parallac = Altitude.above(dec, ha, lat)

    assert alt == 0
    assert az == 0
    assert parallac == 0
