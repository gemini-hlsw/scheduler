# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""RT-27: NightTracker's startup sky computation runs off the event loop."""

import asyncio
import threading
from datetime import datetime, timedelta, UTC
from unittest.mock import patch

import pytest
from lucupy.minimodel import Site

from scheduler.core.events.queue.events import NightEvent
from scheduler.night_monitor.night_tracker import NightTracker


def _future_events(site):
    """Seven canned night events with future times (so _get_correct_events
    keeps them as the upcoming night)."""
    base = datetime.now(UTC) + timedelta(hours=1)
    labels = ["Midnight", "Sunset", "Sunrise", "Evening 12° Twilight",
              "Morning 12° Twilight", "Moonrise", "Moonset"]
    return [NightEvent(description=f"{label} at {site.name}",
                       time=base + timedelta(minutes=10 * i),
                       site=site)
            for i, label in enumerate(labels)]


def test_construction_does_not_compute_sky_events():
    """__init__ must be cheap: no sky.night_events call during construction."""
    with patch.object(NightTracker, "calculate_night_events") as mock_calc:
        tracker = NightTracker(datetime.now(UTC), frozenset([Site.GN]), object())
        mock_calc.assert_not_called()
        assert not tracker.sorted_night_events


@pytest.mark.asyncio
async def test_prepare_computes_events_off_the_loop():
    """prepare() runs the blocking sky computation in a worker thread (not the
    event-loop thread) and populates the sorted event list."""
    loop_thread = threading.current_thread()
    seen_threads = []

    def fake_calc(date, site):
        seen_threads.append(threading.current_thread())
        return _future_events(site)

    with patch.object(NightTracker, "calculate_night_events", side_effect=fake_calc):
        tracker = NightTracker(datetime.now(UTC), frozenset([Site.GN]), object())
        await tracker.prepare()

    assert seen_threads, "calculate_night_events was never called"
    assert all(t is not loop_thread for t in seen_threads), \
        "sky computation ran on the event-loop thread"
    # End-of-night is appended after the per-site events.
    assert tracker.sorted_night_events[-1].description == "End of Night"


@pytest.mark.asyncio
async def test_prepare_is_idempotent():
    """A second prepare() does not recompute the events."""
    with patch.object(NightTracker, "calculate_night_events",
                      side_effect=lambda d, s: _future_events(s)) as mock_calc:
        tracker = NightTracker(datetime.now(UTC), frozenset([Site.GN]), object())
        await tracker.prepare()
        first = tracker.sorted_night_events
        calls_after_first = mock_calc.call_count

        await tracker.prepare()

    assert tracker.sorted_night_events is first
    assert mock_calc.call_count == calls_after_first
