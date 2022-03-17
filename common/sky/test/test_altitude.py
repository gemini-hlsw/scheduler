import pytest
from common.sky.altitude import Altitude


def test_altitude_above_zero():
    """
    Test that altitude is 0 at the equator.
    """

    assert Altitude.above(0) == 0