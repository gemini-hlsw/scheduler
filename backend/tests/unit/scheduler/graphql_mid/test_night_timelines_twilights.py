# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
The UI draws every plan of the night between eveTwilight and mornTwilight, so those must be the real
twilights, not the plan's time grid, which a custom start of night shortens.
"""

from datetime import UTC, datetime

from lucupy.minimodel import NightIndex, Site

from scheduler.core.events.queue import NightlyTimeline
from scheduler.graphql_mid.types import SNightTimelines

NIGHT_IDX = NightIndex(0)
EVE_TWILIGHT = datetime(2026, 10, 29, 5, 0, tzinfo=UTC)
MORN_TWILIGHT = datetime(2026, 10, 29, 15, 30, tzinfo=UTC)
CUSTOM_START = datetime(2026, 10, 29, 9, 0, tzinfo=UTC)
GRID_END = datetime(2026, 10, 29, 15, 29, tzinfo=UTC)


def _timeline() -> NightlyTimeline:
    timeline = NightlyTimeline()
    timeline.stitched_timeline = {NIGHT_IDX: {Site.GN: []}}
    timeline.set_night_length(NIGHT_IDX, Site.GN, CUSTOM_START, GRID_END)
    return timeline


def _site_entries(timeline: NightlyTimeline):
    (night,) = SNightTimelines.from_computed_stitched_timelines(timeline).night_timeline
    (site_entries,) = night.time_entries_by_site
    return site_entries


def test_twilights_do_not_move_with_a_custom_start_of_night():
    timeline = _timeline()
    timeline.set_twilights(NIGHT_IDX, Site.GN, EVE_TWILIGHT, MORN_TWILIGHT)
    site_entries = _site_entries(timeline)
    assert site_entries.eve_twilight == EVE_TWILIGHT
    assert site_entries.morn_twilight == MORN_TWILIGHT


def test_twilights_fall_back_to_the_time_grid_when_not_recorded():
    site_entries = _site_entries(_timeline())
    assert site_entries.eve_twilight == CUSTOM_START
    assert site_entries.morn_twilight == GRID_END


def test_custom_night_times_are_sent_when_recorded():
    timeline = _timeline()
    timeline.set_custom_night_times(NIGHT_IDX, Site.GN, CUSTOM_START, None)
    site_entries = _site_entries(timeline)
    assert site_entries.night_start == CUSTOM_START
    assert site_entries.night_end is None


def test_no_custom_night_times_by_default():
    site_entries = _site_entries(_timeline())
    assert site_entries.night_start is None
    assert site_entries.night_end is None
