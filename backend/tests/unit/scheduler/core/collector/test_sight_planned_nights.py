# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

"""The Sight loader fetches Stage 1 arrays and builds target info only for the
nights the Selector plans (num_of_nights: one in realtime, the full range in
validation), while Stage 2 still covers the whole visibility window."""

from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lucupy.minimodel import NightIndex, ObservationID, Site

from scheduler.core.components.collector import collector as collector_mod
from scheduler.core.components.collector.collector import Collector

START = date(2026, 10, 7)


class _Obs:
    def __init__(self, obs_id: str, target: str, site: Site = Site.GN):
        self.id = ObservationID(obs_id)
        self.site = site
        self._base = SimpleNamespace(name=target)

    def base_target(self):
        return self._base

    def exec_time(self):
        return timedelta(hours=1)

    def total_used(self):
        return timedelta(0)


def _collector(num_of_nights, num_nights_calculated: int) -> Collector:
    obj = Collector.__new__(Collector)
    obj.num_of_nights = num_of_nights
    obj.num_nights_calculated = num_nights_calculated
    obj.time_grid = [
        SimpleNamespace(to_datetime=lambda d=START + timedelta(days=i): datetime(d.year, d.month, d.day))
        for i in range(num_nights_calculated)
    ]
    obj._target_info = {}
    obj._observations = {}
    return obj


@pytest.mark.parametrize("num_of_nights, expected", [
    (1, {0}),            # realtime
    (3, {0, 1, 2}),      # validation over three nights
    (0, set()),
])
def test_planned_nights(num_of_nights, expected):
    assert _collector(num_of_nights, 5)._planned_nights() == {NightIndex(i) for i in expected}


def _visible(*obs_ids):
    return [SimpleNamespace(observation_id=o, visible_ranges=[[0, 3]], remaining_minutes=30) for o in obs_ids]


@asynccontextmanager
async def _session():
    yield object()


async def _fetch(obj, filtered, visible_per_night):
    calc = MagicMock()
    calc.get_visible_observations = AsyncMock(side_effect=visible_per_night)
    calc.get_stage1_greedymax_bulk = AsyncMock(return_value={})
    with patch.object(collector_mod, "session_scope", _session), \
            patch.object(collector_mod, "Calculator", return_value=calc), \
            patch.object(collector_mod, "_obs_to_request", side_effect=lambda o: o):
        per_night = await obj._fetch_sight_data(filtered, START, START + timedelta(days=2), ["GN"])
    return calc, per_night


@pytest.mark.asyncio
async def test_stage1_fetch_stops_at_last_planned_night():
    obj = _collector(num_of_nights=1, num_nights_calculated=3)
    tonight, later = _Obs("o1", "T1"), _Obs("o2", "T2")
    filtered = {NightIndex(0): [tonight, later], NightIndex(1): [tonight, later], NightIndex(2): [later]}

    calc, per_night = await _fetch(obj, filtered, [_visible("o1"), _visible("o1", "o2"), _visible("o2")])

    # Stage 2 is still read for every night: the remaining-time sums need them.
    assert calc.get_visible_observations.await_count == 3
    # o1's cumulative remaining minutes on night 0 include nights 0 and 1.
    assert per_night[NightIndex(0)][2]["o1"] == 60
    # Stage 1: only targets visible on the planned night, only up to that night.
    names, sites, start, end = calc.get_stage1_greedymax_bulk.await_args.args
    assert names == ["T1"]
    assert (start, end) == (START, START)


@pytest.mark.asyncio
async def test_no_planned_nights_skips_stage1_fetch():
    obj = _collector(num_of_nights=0, num_nights_calculated=2)
    obs = _Obs("o1", "T1")
    filtered = {NightIndex(0): [obs], NightIndex(1): [obs]}

    calc, per_night = await _fetch(obj, filtered, [_visible("o1"), _visible("o1")])

    calc.get_stage1_greedymax_bulk.assert_not_awaited()
    assert per_night[NightIndex(0)][1] == {}


def test_target_info_built_only_for_planned_nights():
    obj = _collector(num_of_nights=1, num_nights_calculated=2)
    obj.night_events = {Site.GN: SimpleNamespace(times=[[0] * 10, [0] * 10])}
    tonight, later = _Obs("o1", "T1"), _Obs("o2", "T2")
    filtered = {NightIndex(0): [tonight, later], NightIndex(1): [tonight, later]}
    stage1 = {"T1": {"nights": {f"GN_{START.isoformat()}": {}}}}
    per_night = {
        NightIndex(0): ({"o1"}, stage1, {"o1": 60}, {"o1": [[0, 3]]}),
        NightIndex(1): ({"o1", "o2"}, stage1, {"o1": 30, "o2": 30}, {"o1": [[0, 3]], "o2": [[0, 3]]}),
    }

    with patch.object(collector_mod, "build_target_info", side_effect=lambda *a: SimpleNamespace()) as build:
        obj._apply_sight_visibility(filtered, per_night)

    assert build.call_count == 1
    assert set(obj._target_info[("T1", tonight.id)]) == {NightIndex(0)}
    # Visible only on an unplanned night: still registered, with no target info,
    # so the Selector skips it quietly instead of warning about a missing entry.
    assert obj._target_info[("T2", later.id)] == {}
    assert obj._observations[later.id] == (later, later.base_target())
    # Visible observations are still published for every night.
    assert set(obj._visible_obs_by_night) == {NightIndex(0), NightIndex(1)}
