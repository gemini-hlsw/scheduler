# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""The custom night window is part of the cache identity.

It comes from the build parameters, so it varies per run over the same dates. Leaving it
out of the key made two runs share one slot: the entry was rebuilt on every alternating
call, running the astropy sky math the cache exists to avoid, and a caller that re-read
its night events got the other run's window.
"""

from datetime import datetime, timedelta

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time, TimeDelta
from lucupy.minimodel import Site

from scheduler.core.components.nighteventsmanager import NightEventsManager

_SITE = Site.GN
_SLOT = TimeDelta(1.0 * u.min)
# GN twilights for this grid are 05:32 -> 15:17 UT, so both custom windows sit inside the
# night; NightEvents only applies a window that falls between the real twilights.
_NIGHT = datetime(2026, 8, 26)
_EARLY = Time('2026-08-26 06:00:00', format='iso', scale='utc')
_LATE = Time('2026-08-26 08:00:00', format='iso', scale='utc')


def _grid() -> Time:
    return Time(np.arange(_NIGHT, _NIGHT + timedelta(days=1.0), timedelta(days=1.0)))


@pytest.fixture(autouse=True)
def _clean_cache():
    NightEventsManager._night_events = {}
    yield
    NightEventsManager._night_events = {}


def _get(start, end=None):
    return NightEventsManager.get_night_events(_grid(), start, end, _SLOT, _SITE)


def test_the_same_request_is_cached():
    assert _get(None) is _get(None), "an identical request must not recompute"


def test_a_fresh_grid_object_still_hits_the_cache():
    # time_grid[0] is a new Time object on every call, so the key relies on Time hashing
    # by value. If that ever changed, every call would miss and rebuild.
    first = _get(None)
    assert NightEventsManager.get_night_events(_grid(), None, None, _SLOT, _SITE) is first


def test_different_night_windows_get_different_entries():
    default, early, late = _get(None), _get(_EARLY), _get(_LATE)

    assert early is not default
    assert late is not early
    assert len({id(default), id(early), id(late)}) == 3
    assert len(NightEventsManager._night_events) == 3


def test_windows_do_not_evict_each_other():
    """The regression: alternating between two windows used to rebuild on every call.

    Both requests shared one slot, so each call found the other's window cached, replaced
    the entry and re-ran sky.night_events.
    """
    early_first = _get(_EARLY)
    _get(_LATE)

    assert _get(_EARLY) is early_first, "the second window evicted the first"


def test_each_entry_keeps_its_own_window():
    early, late = _get(_EARLY), _get(_LATE)

    assert early.night_start_time == _EARLY
    assert late.night_start_time == _LATE
    # And the earlier handle is not retroactively pointed at the later window.
    assert _get(_EARLY).night_start_time == _EARLY
