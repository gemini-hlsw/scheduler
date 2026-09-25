# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

"""Elevation constraints in Stage 2.

Hour angle constraints are in hours and also cap the airmass at ElevationLimits.AIRMASS_LIMIT.
"""

from datetime import datetime, timedelta, timezone

import numpy as np

from scheduler.services.sight.calculations.arrays import pack_array
from scheduler.services.sight.calculations.stage2 import (
    ElevationType,
    ObservationConstraints,
    TimingWindow,
    calculate_visibility,
)

NIGHT_START = datetime(2026, 6, 26, 6, 0, tzinfo=timezone.utc)


def _visibility(hourangle_hours: np.ndarray, airmass: np.ndarray, constraints: ObservationConstraints):
    n = len(hourangle_hours)
    zeros = pack_array(np.zeros(n))
    return calculate_visibility(
        alt_bytes=pack_array(np.radians(np.full(n, 70.0))),
        az_bytes=zeros,
        airmass_bytes=pack_array(airmass),
        hourangle_bytes=pack_array(np.radians(hourangle_hours * 15.0)),
        ra_bytes=pack_array(np.full(n, 150.0)),
        dec_bytes=pack_array(np.full(n, -30.0)),
        sun_alt_bytes=pack_array(np.radians(np.full(n, -30.0))),
        moon_alt_bytes=pack_array(np.radians(np.full(n, -45.0))),
        moon_ra_bytes=pack_array(np.full(n, 330.0)),
        moon_dec_bytes=pack_array(np.full(n, 10.0)),
        sun_moon_ang_bytes=pack_array(np.radians(np.full(n, 180.0))),
        moon_dist_bytes=pack_array(np.full(n, 384_400_000.0)),
        night_start=NIGHT_START,
        night_duration_minutes=n,
        constraints=constraints,
    )


def _ha_constraints(ha_min: float, ha_max: float) -> ObservationConstraints:
    return ObservationConstraints(
        elevation_type=ElevationType.HOUR_ANGLE,
        elevation_min=ha_min,
        elevation_max=ha_max,
        timing_windows=[TimingWindow(start=NIGHT_START, end=NIGHT_START + timedelta(days=1))],
    )


def test_hour_angle_constraints_are_in_hours():
    hourangle = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    res = _visibility(hourangle, np.full(len(hourangle), 1.2), _ha_constraints(-1.0, 2.0))
    assert hourangle[res.visibility_mask].tolist() == [-1.0, 0.0, 1.0, 2.0]


def test_hour_angle_constraints_cap_airmass():
    hourangle = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    airmass = np.array([2.5, 1.5, 1.1, 1.5, 2.2])
    res = _visibility(hourangle, airmass, _ha_constraints(-5.0, 5.0))
    assert hourangle[res.visibility_mask].tolist() == [-2.0, 0.0, 2.0, 4.0]
