# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

"""Regression tests for Stage-2 sky-brightness handling.

The Sight rewrite called lucupy's *array-based* ``calculate_sky_brightness``
once per time slot with scalar arguments inside a loop, wrapped in a bare
``except`` that defaulted every slot to ``SBANY``. The result was that every
observation requiring SB < Any (SB20/SB50/SB80, i.e. dark/gray) got zero visible
minutes and was silently dropped from scheduling. These tests pin the corrected,
vectorised behaviour.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from lucupy.minimodel import SkyBackground

from scheduler.services.sight.calculations.stage2 import (
    ObservationConstraints,
    _calculate_sky_brightness_array,
    calculate_visibility,
)
from scheduler.services.sight.calculations.arrays import pack_array


# Mean Earth-Moon distance in meters (~0.00257 AU). The moon distance is
# per-slot in METERS, as lucupy's accurate_location returns it: lucupy
# normalizes by EQUAT_RAD (meters), and an AU Quantity does not simplify.
_MOON_DIST_M = 384_400_000.0


def _moon_down_inputs(n: int, sun_alt_deg: float):
    """Arrays for a target overhead with the moon well below the horizon."""
    return dict(
        ra=np.full(n, 150.0),                 # degrees
        dec=np.full(n, -30.0),                # degrees
        alt=np.radians(np.full(n, 70.0)),     # radians (target high up)
        sun_alt=np.radians(np.full(n, sun_alt_deg)),
        moon_alt=np.radians(np.full(n, -45.0)),  # moon below horizon -> dark
        moon_ra=np.full(n, 330.0),            # degrees (far from target/sun)
        moon_dec=np.full(n, 10.0),            # degrees
        sun_moon_ang=np.radians(np.full(n, 180.0)),
        moon_dist_m=np.full(n, _MOON_DIST_M),  # meters, per-slot
    )


def test_sky_brightness_not_uniformly_sbany():
    """The core regression: dark conditions must NOT collapse to SBANY."""
    sb = _calculate_sky_brightness_array(**_moon_down_inputs(30, sun_alt_deg=-30.0))

    assert sb.dtype == np.float64
    assert np.all(np.isfinite(sb))
    # Moon well below the horizon + deep night => darkest sky (SB20).
    assert not np.all(sb == float(SkyBackground.SBANY)), (
        "sky brightness collapsed to SBANY everywhere (the original bug)"
    )
    assert np.allclose(sb, float(SkyBackground.SB20))


def test_sky_brightness_handles_twilight_without_raising():
    """Twilight slots reach lucupy's ztwilight (needs Angle, not Quantity).

    This is the exact branch the scalar/Quantity misuse blew up on.
    """
    # Sun between the -12 and -18.5 twilight band exercises ztwilight.
    sb = _calculate_sky_brightness_array(**_moon_down_inputs(15, sun_alt_deg=-15.0))
    assert np.all(np.isfinite(sb))
    assert np.all(sb <= float(SkyBackground.SBANY))


def test_bright_moon_overhead_is_brighter_than_dark():
    """A bright moon close to the target yields a brighter sky than moon-down."""
    n = 20
    dark = _calculate_sky_brightness_array(**_moon_down_inputs(n, sun_alt_deg=-30.0))

    bright_inputs = _moon_down_inputs(n, sun_alt_deg=-30.0)
    bright_inputs.update(
        moon_alt=np.radians(np.full(n, 60.0)),  # moon high
        moon_ra=np.full(n, 150.0),              # near the target
        moon_dec=np.full(n, -30.0),
        sun_moon_ang=np.radians(np.full(n, 10.0)),  # near full moon
    )
    bright = _calculate_sky_brightness_array(**bright_inputs)

    assert np.all(bright >= dark)
    assert bright.max() > dark.max()


def test_calculate_visibility_keeps_dark_constrained_observation():
    """End-to-end: an SB50 observation keeps visible minutes on a dark night."""
    n = 120
    ne = _moon_down_inputs(n, sun_alt_deg=-30.0)

    common = dict(
        alt_bytes=pack_array(ne["alt"]),
        az_bytes=pack_array(np.full(n, 0.0)),
        airmass_bytes=pack_array(np.full(n, 1.2)),  # within default airmass range
        hourangle_bytes=pack_array(np.zeros(n)),
        ra_bytes=pack_array(ne["ra"]),
        dec_bytes=pack_array(ne["dec"]),
        sun_alt_bytes=pack_array(ne["sun_alt"]),
        moon_alt_bytes=pack_array(ne["moon_alt"]),
        moon_ra_bytes=pack_array(ne["moon_ra"]),
        moon_dec_bytes=pack_array(ne["moon_dec"]),
        sun_moon_ang_bytes=pack_array(ne["sun_moon_ang"]),
        moon_dist_bytes=pack_array(ne["moon_dist_m"]),
        night_start=datetime(2026, 6, 26, 0, 0, tzinfo=timezone.utc),
        night_duration_minutes=n,
    )

    dark = calculate_visibility(
        constraints=ObservationConstraints(target_sb=0.5, has_resources=True, can_schedule=True),
        **common,
    )
    assert dark.remaining_minutes > 0
    assert dark.sky_brightness is not None
    assert not all(v == float(SkyBackground.SBANY) for v in dark.sky_brightness)


def test_moon_up_far_separation_not_all_sbany():
    """Moon-up slots must not uniformly classify as SBANY.

    Regression for the AU-unit bug: the moon distance was passed as an AU
    Quantity, lucupy's ``earth_moon_dist / (60.27 * EQUAT_RAD)`` (meters) kept
    an un-simplified AU/m composite (raw value ~7e-12 instead of ~1.0), and
    assigning the result into a plain ndarray stripped the unit — inflating the
    lunar term by ~1e22 so EVERY slot with the moon above the horizon computed
    as brighter than SBANY thresholds. A crescent moon at moderate altitude,
    120 degrees from the target, must yield a sky darker than SBANY.
    """
    n = 30
    inputs = _moon_down_inputs(n, sun_alt_deg=-30.0)
    inputs.update(
        moon_alt=np.radians(np.full(n, 30.0)),      # moon up
        moon_ra=np.full(n, 270.0),                  # ~120 deg from target
        moon_dec=np.full(n, 10.0),
        sun_moon_ang=np.radians(np.full(n, 60.0)),  # phase 120 deg -> crescent
    )
    sb = _calculate_sky_brightness_array(**inputs)
    assert np.all(np.isfinite(sb))
    assert not np.all(sb == float(SkyBackground.SBANY)), (
        "every moon-up slot classified SBANY (the AU-unit bug)"
    )
