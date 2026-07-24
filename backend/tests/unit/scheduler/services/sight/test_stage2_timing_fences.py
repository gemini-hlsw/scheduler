# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

"""Timing-window fence semantics in Stage 2.

The legacy calculator kept a slot iff ``window.start <= t_k <= window.end``
(inclusive both ends, comparing actual slot times). The first Sight port
floored both fences into minute indices and sliced end-exclusive, which always
dropped the final in-window slot and could admit a partial slot before the
window opened. These tests pin the restored inclusive semantics.
"""

from datetime import datetime, timedelta, timezone

import numpy as np

from scheduler.services.sight.calculations.arrays import pack_array
from scheduler.services.sight.calculations.stage2 import (
    ObservationConstraints,
    TimingWindow,
    calculate_visibility,
)
from scheduler.services.sight.helpers import align_to_start

NIGHT_START = datetime(2026, 6, 26, 6, 0, tzinfo=timezone.utc)


def _visibility_for_window(n: int, tw_start: datetime, tw_end: datetime):
    """All non-timing filters pass; the mask is purely the timing window."""
    zeros = pack_array(np.zeros(n))
    return calculate_visibility(
        alt_bytes=pack_array(np.radians(np.full(n, 70.0))),
        az_bytes=zeros,
        airmass_bytes=pack_array(np.full(n, 1.2)),
        hourangle_bytes=zeros,
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
        constraints=ObservationConstraints(
            timing_windows=[TimingWindow(start=tw_start, end=tw_end)],
            has_resources=True,
            can_schedule=True,
        ),
    )


def test_window_on_slot_boundaries_is_inclusive_both_ends():
    """[start+10min, start+20min] keeps slots 10..20 (11 slots), as legacy."""
    res = _visibility_for_window(
        60,
        NIGHT_START + timedelta(minutes=10),
        NIGHT_START + timedelta(minutes=20),
    )
    visible = np.where(res.visibility_mask)[0]
    assert visible.tolist() == list(range(10, 21))
    assert res.remaining_minutes == 11


def test_mid_minute_fences_use_ceil_start_floor_end():
    """Start 10m30s -> first slot 11; end 20m30s -> last slot 20."""
    res = _visibility_for_window(
        60,
        NIGHT_START + timedelta(minutes=10, seconds=30),
        NIGHT_START + timedelta(minutes=20, seconds=30),
    )
    visible = np.where(res.visibility_mask)[0]
    assert visible.tolist() == list(range(11, 21))


def test_window_spanning_night_end_is_clamped():
    res = _visibility_for_window(
        30,
        NIGHT_START + timedelta(minutes=25),
        NIGHT_START + timedelta(hours=5),
    )
    visible = np.where(res.visibility_mask)[0]
    assert visible.tolist() == list(range(25, 30))


def test_window_before_night_yields_nothing():
    res = _visibility_for_window(
        30,
        NIGHT_START - timedelta(hours=2),
        NIGHT_START - timedelta(minutes=1),
    )
    assert res.remaining_minutes == 0


def test_align_to_start_shifts_and_pads():
    arr = np.arange(10)
    assert align_to_start(arr, 0).tolist() == list(range(10))
    assert align_to_start(arr, 3).tolist() == list(range(3, 10))
    assert align_to_start(arr, -2).tolist() == [0, 0] + list(range(10))
