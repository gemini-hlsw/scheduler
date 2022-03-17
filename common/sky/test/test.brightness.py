import pytest
from common.sky.brightness import calculate_sky_brightness, calculate_sky_brightness_qpt, 
from common.sky.brightness import convert_to_sky_background

def test_sky_brigthness_qpt():
    """
    Test that the sky brightness is calculated correctly.
    """

    assert calculate_sky_brightness_qpt(1, 1, 1) == 1


def test_sky_brigthness():
    """
    Test that the sky brightness is calculated correctly.
    """

    assert calculate_sky_brightness(1, 1, 1) == 1

def test_convert_to_sky_background():
    """
    Test that the sky brightness is calculated correctly.
    """

    assert convert_to_sky_background(1) == 1
