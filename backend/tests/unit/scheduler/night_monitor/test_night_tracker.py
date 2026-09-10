# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""The night events must be computed off the event loop.

`sky.night_events` is heavy synchronous math. It used to run in NightTracker.__init__,
which NightMonitor calls on the loop that also owns the ODB subscription -- stalling it at
startup, before the first plan is even requested.
"""

import threading
from datetime import datetime, UTC
from unittest.mock import AsyncMock, patch

import pytest
from lucupy.minimodel import Site

from scheduler.night_monitor.night_tracker import NightTracker

_SITES = frozenset({Site.GN})
_DATE = datetime(2026, 8, 26, tzinfo=UTC)


def _tracker() -> NightTracker:
    return NightTracker(_DATE, _SITES, AsyncMock())


def test_construction_computes_nothing():
    # __init__ runs on the loop, so it must not do the sky math.
    with patch.object(NightTracker, "_compute_sorted_night_events") as compute:
        tracker = _tracker()

    compute.assert_not_called()
    assert tracker.sorted_night_events == []


def test_str_survives_an_unprepared_tracker():
    # __str__ iterates the events; deferring them must not break logging.
    assert isinstance(str(_tracker()), str)


@pytest.mark.asyncio
async def test_prepare_populates_the_events():
    tracker = _tracker()
    await tracker.prepare()

    assert tracker.sorted_night_events, "prepare must compute the night events"
    # The end-of-night sentinel is appended last and is what ends the night loop.
    assert tracker.sorted_night_events[-1].description == "End of Night"
    times = [e.time for e in tracker.sorted_night_events]
    assert times == sorted(times), "events must be in chronological order"


@pytest.mark.asyncio
async def test_prepare_runs_off_the_event_loop():
    tracker = _tracker()
    ran_on = {}

    def record():
        ran_on["thread"] = threading.get_ident()
        return [object()]

    with patch.object(tracker, "_compute_sorted_night_events", record):
        await tracker.prepare()

    assert ran_on["thread"] != threading.get_ident(), \
        "the sky math must not run on the loop's thread"


@pytest.mark.asyncio
async def test_prepare_is_idempotent():
    # start_tracking calls prepare, and a retry must not recompute.
    tracker = _tracker()
    await tracker.prepare()
    first = tracker.sorted_night_events

    with patch.object(NightTracker, "_compute_sorted_night_events") as compute:
        await tracker.prepare()

    compute.assert_not_called()
    assert tracker.sorted_night_events is first


@pytest.mark.asyncio
async def test_start_tracking_prepares_first():
    # Nothing else calls prepare, so start_tracking owns it: without this the tracker
    # would iterate an empty event list and the night would never advance.
    tracker = _tracker()
    with patch.object(tracker, "prepare", AsyncMock()) as prepare:
        await tracker.start_tracking()

    prepare.assert_awaited_once()
