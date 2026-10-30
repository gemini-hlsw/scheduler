# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import UTC, datetime, timedelta

import astropy.units as u
import numpy as np
from astropy.time import Time, TimeDelta
from lucupy.minimodel import Site

from scheduler.core.calculations.moon import moon_altitudes, moon_phase
from scheduler.core.components.nighteventsmanager import NightEventsManager


def test_matches_the_moon_altitude_the_scheduler_uses():
    """Same altitudes as NightEvents over its own time grid."""
    time_grid = Time(['2026-10-29 00:00'])
    night_events = NightEventsManager.get_night_events(time_grid, None, None, TimeDelta(1 * u.min), Site.GN)
    grid_times = night_events.utc_times[0]
    expected = night_events.moon_alt[0].deg

    start, end = grid_times[0], grid_times[-1]
    computed = moon_altitudes(Site.GN, start, end, step=timedelta(minutes=1))

    assert [t for t, _ in computed] == list(grid_times)
    np.testing.assert_allclose([alt for _, alt in computed], expected, atol=0.01)


def test_includes_both_ends():
    start = datetime(2026, 10, 29, 4, 0, tzinfo=UTC)
    computed = moon_altitudes(Site.GS, start, start + timedelta(hours=1))
    assert len(computed) == 13
    assert computed[-1][0] == start + timedelta(hours=1)


def test_phase_around_the_full_moon_of_26_october_2026():
    illumination, _ = moon_phase(Site.GN, datetime(2026, 10, 26, 10, 0, tzinfo=UTC))
    assert illumination > 0.99


def test_phase_waning_then_waxing_around_the_new_moon_of_9_november_2026():
    before, waxing_before = moon_phase(Site.GN, datetime(2026, 11, 5, 10, 0, tzinfo=UTC))
    after, waxing_after = moon_phase(Site.GN, datetime(2026, 11, 13, 10, 0, tzinfo=UTC))
    assert 0.05 < before < 0.3 and not waxing_before
    assert 0.05 < after < 0.3 and waxing_after
