# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime, timedelta

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time, TimeDelta
from lucupy.minimodel import Site

from scheduler.core.components.nighteventsmanager import NightEventsManager

_SITE = Site.GN
_SLOT = TimeDelta(1.0 * u.min)
# GN twilights for this grid are 05:32 -> 15:17 UT.
_NIGHT = datetime(2026, 8, 26)
_INSIDE_EARLY = Time('2026-08-26 06:00:00', format='iso', scale='utc')
_INSIDE_LATE = Time('2026-08-26 08:00:00', format='iso', scale='utc')
_BEFORE_EVENING_TWILIGHT = Time('2026-08-26 04:00:00', format='iso', scale='utc')
# The whole-day slip: the operator names the evening of the *next* night.
_NEXT_NIGHT = Time('2026-08-27 06:00:00', format='iso', scale='utc')


def _grid() -> Time:
    return Time(np.arange(_NIGHT, _NIGHT + timedelta(days=1.0), timedelta(days=1.0)))


@pytest.fixture(autouse=True)
def _clean_cache():
    NightEventsManager._night_events = {}
    yield
    NightEventsManager._night_events = {}


def _get(start=None, end=None):
    return NightEventsManager.get_night_events(_grid(), start, end, _SLOT, _SITE)


def test_a_window_inside_the_twilights_still_applies():
    night_events = _get(_INSIDE_EARLY, _INSIDE_LATE)

    assert night_events.times[0][0] == _INSIDE_EARLY
    # times stops one slot short of the end, so check the window length instead.
    assert night_events.num_timeslots_per_night[0] == 120


def test_a_night_start_before_evening_twilight_is_refused():
    with pytest.raises(ValueError, match='night start'):
        _get(start=_BEFORE_EVENING_TWILIGHT)


def test_a_night_start_on_the_following_night_is_refused():
    with pytest.raises(ValueError) as exc:
        _get(start=_NEXT_NIGHT)

    # The message has to name the night actually built, or the operator cannot tell
    # which of the two local dates they were supposed to pick.
    assert '2026-08-27 06:00' in str(exc.value)
    assert '2026-08-26 05:3' in str(exc.value)
    assert '2026-08-26 15:1' in str(exc.value)


def test_a_night_end_after_morning_twilight_is_refused():
    with pytest.raises(ValueError, match='night end'):
        _get(end=Time('2026-08-26 23:00:00', format='iso', scale='utc'))


def test_a_night_end_before_the_custom_start_is_refused():
    with pytest.raises(ValueError, match='night end'):
        _get(start=_INSIDE_LATE, end=_INSIDE_EARLY)


def test_a_refused_window_is_not_cached():
    with pytest.raises(ValueError):
        _get(start=_NEXT_NIGHT)

    assert NightEventsManager._night_events == {}, "a failed build must not leave an entry"
