import pytest
from common.sky.brightness import calculate_sky_brightness, calculate_sky_brightness_qpt
from common.sky.brightness import convert_to_sky_background
from common.sky.constants import KZEN
from astropy.coordinates import SkyCoord, Angle
from astropy.time import Time
import astropy.units as u


def test_sky_brigthness_qpt():
    """
    Test that the sky brightness is calculated correctly according to the QPT method.
    """
    ...


def test_sky_brigthness():
    """
    Test that the sky brightness is calculated correctly.
    """
    ...


def test_convert_to_sky_background():
    """
    Test the conversion to sky background.
    """
    ...
