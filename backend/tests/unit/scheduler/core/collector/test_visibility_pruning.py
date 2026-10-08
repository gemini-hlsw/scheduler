# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Observations not visible on any scheduled night skip the rest of the visibility period.

They are never scored, so neither the night configuration filter, the visibility queries,
nor the Stage-1 fetch run for them past the scheduled nights. The observations that are
kept must get exactly the remaining visibility fraction they got before.
"""

from collections import namedtuple
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from lucupy.minimodel import NightIndex, Site

from scheduler.core.components.collector import collector as collector_mod
from scheduler.core.components.collector.collector import Collector

_SITE = Site.GS
_START = datetime(2026, 2, 1, tzinfo=timezone.utc)
_ObsId = namedtuple('_ObsId', 'id')

# Visible minutes per observation and night. TONIGHT is visible tonight; LATER only afterwards.
_MINUTES = {
    'TONIGHT': [10, 20, 30],
    'LATER': [0, 40, 50],
}


def _obs(obs_id: str) -> MagicMock:
    obs = MagicMock()
    obs.id = _ObsId(obs_id)
    obs.site = _SITE
    obs.base_target.return_value = SimpleNamespace(name=f'target-{obs_id}')
    obs.exec_time.return_value = timedelta(minutes=30)
    obs.total_used.return_value = timedelta(0)
    return obs


def _night_date(n: int):
    return (_START + timedelta(days=n)).date()


def _collector(num_of_nights: int = 1) -> Collector:
    collector = Collector.__new__(Collector)
    collector.num_of_nights = num_of_nights
    collector.num_nights_calculated = len(_MINUTES['TONIGHT'])
    collector.sites = frozenset({_SITE})
    collector.end_vis_time = _START + timedelta(days=collector.num_nights_calculated - 1)
    collector.time_grid = [SimpleNamespace(to_datetime=lambda n=n: _START + timedelta(days=n))
                           for n in range(collector.num_nights_calculated)]
    collector.get_program = MagicMock(return_value=SimpleNamespace(start=None, end=None))
    collector._target_info = {}
    collector._visible_obs_by_night = {}
    collector._observations = {}
    return collector


def _record_filter(collector: Collector) -> dict:
    """Replace the night configuration filter with one that passes everything and records its input."""
    filtered_ids: dict = {}

    def filter_nights(observations, _nc, nights):
        for n in nights:
            filtered_ids[n] = sorted(o.id.id for _p, o in observations)
        return {NightIndex(n): [o for _p, o in observations] for n in nights}

    collector._filter_nights = filter_nights
    return filtered_ids


class _FakeCalculator:
    # (observation, night) pairs with no stored row: only get_visible_observations,
    # which calculates them, returns their minutes.
    unstored: set = set()

    def __init__(self, _session):
        self.queried: dict = {}
        self.range_queries: list = []
        self.stage1_targets = None
        _FakeCalculator.last = self

    async def get_remaining_minutes_in_range(self, observation_ids, start, end):
        nights = [i for i in range(len(_MINUTES['TONIGHT'])) if start <= _night_date(i) <= end]
        self.range_queries.append((sorted(observation_ids), nights))
        return {(obs_id, _night_date(n)): _MINUTES[obs_id][n]
                for obs_id in observation_ids for n in nights if (obs_id, n) not in self.unstored}

    async def get_visible_observations(self, requests, night_date):
        n = next(i for i in range(len(_MINUTES['TONIGHT'])) if _night_date(i) == night_date)
        self.queried[n] = sorted(requests)
        return [SimpleNamespace(observation_id=obs_id, visible_ranges=[(0, 1)],
                                remaining_minutes=_MINUTES[obs_id][n])
                for obs_id in requests if _MINUTES[obs_id][n] > 0]

    async def get_stage1_greedymax_bulk(self, target_names, *_args):
        self.stage1_targets = target_names
        return {}


@asynccontextmanager
async def _session():
    yield None


@pytest.fixture
def sight():
    _FakeCalculator.unstored = set()
    with patch.object(collector_mod, 'session_scope', _session), \
            patch.object(collector_mod, 'Calculator', _FakeCalculator), \
            patch.object(collector_mod, '_obs_to_request', lambda o: o.id.id):
        yield


@pytest.mark.asyncio
async def test_sight_skips_the_remaining_nights_of_observations_not_visible_tonight(sight):
    collector = _collector()
    filtered_ids = _record_filter(collector)
    parsed = [('p', _obs('TONIGHT')), ('p', _obs('LATER'))]

    filtered, per_night = await collector._fetch_sight_data_for_scheduled_nights(
        parsed, {}, _START.date(), _night_date(2), ['GS'])

    calc = _FakeCalculator.last
    assert filtered_ids == {0: ['LATER', 'TONIGHT'], 1: ['TONIGHT'], 2: ['TONIGHT']}
    # Tonight per night (it needs the visible ranges); the later nights in one range query.
    assert calc.queried == {0: ['LATER', 'TONIGHT']}
    assert calc.range_queries == [(['TONIGHT'], [1, 2])]
    assert calc.stage1_targets == ['target-TONIGHT']
    assert {int(n): [o.id.id for o in obs] for n, obs in filtered.items()} == \
        {0: ['TONIGHT', 'LATER'], 1: ['TONIGHT'], 2: ['TONIGHT']}
    assert {int(n): sorted(data[0]) for n, data in per_night.items()} == \
        {0: ['TONIGHT'], 1: ['TONIGHT'], 2: ['TONIGHT']}


@pytest.mark.asyncio
async def test_sight_remaining_visibility_of_kept_observations_is_unchanged(sight):
    parsed = [('p', _obs('TONIGHT')), ('p', _obs('LATER'))]

    pruned = _collector()
    _record_filter(pruned)
    _, pruned_per_night = await pruned._fetch_sight_data_for_scheduled_nights(
        parsed, {}, _START.date(), _night_date(2), ['GS'])

    # Every night over every observation, as before the pruning.
    unpruned = _collector()
    every_night = {NightIndex(n): [o for _p, o in parsed] for n in range(3)}
    unpruned_per_night = await unpruned._fetch_sight_data(every_night, _START.date(), _night_date(2), ['GS'])

    for n in range(3):
        assert pruned_per_night[NightIndex(n)][2]['TONIGHT'] == unpruned_per_night[NightIndex(n)][2]['TONIGHT']
    assert pruned_per_night[NightIndex(0)][2]['TONIGHT'] == sum(_MINUTES['TONIGHT'])


@pytest.mark.asyncio
async def test_sight_calculates_later_nights_with_no_stored_row(sight):
    _FakeCalculator.unstored = {('TONIGHT', 2)}
    parsed = [('p', _obs('TONIGHT')), ('p', _obs('LATER'))]

    collector = _collector()
    _record_filter(collector)
    _, per_night = await collector._fetch_sight_data_for_scheduled_nights(
        parsed, {}, _START.date(), _night_date(2), ['GS'])

    calc = _FakeCalculator.last
    # Only the missing pair goes through the per-night query, which calculates it.
    assert calc.queried == {0: ['LATER', 'TONIGHT'], 2: ['TONIGHT']}
    assert per_night[NightIndex(2)][0] == {'TONIGHT'}
    assert per_night[NightIndex(0)][2]['TONIGHT'] == sum(_MINUTES['TONIGHT'])


@pytest.mark.asyncio
async def test_sight_with_nothing_visible_tonight_queries_no_remaining_night(sight):
    collector = _collector()
    _record_filter(collector)

    filtered, per_night = await collector._fetch_sight_data_for_scheduled_nights(
        [('p', _obs('LATER'))], {}, _START.date(), _night_date(2), ['GS'])

    calc = _FakeCalculator.last
    assert calc.queried == {0: ['LATER']}
    assert calc.stage1_targets is None
    assert all(not data[0] for data in per_night.values())


@pytest.fixture
def local_stages():
    """Stand-ins for the Sight calculations, keyed by (target, night) through the Stage-1 output."""
    visibility_calls = []

    def night_events(_site, night_date):
        n = next(i for i in range(3) if _night_date(i) == night_date)
        return SimpleNamespace(night=n, night_start=_START + timedelta(days=n), night_duration_minutes=4,
                               sun_alt=None, moon_alt=None, moon_ra=None, moon_dec=None,
                               sun_moon_ang=None, moon_dist=None)

    def stage1(target, _site, ne):
        key = (target.name.removeprefix('target-'), ne.night)
        return SimpleNamespace(ra=key, dec=key, alt=key, az=key, hourangle=key, airmass=key)

    def visibility(**kwargs):
        obs_id, n = kwargs['alt_bytes']
        visibility_calls.append((obs_id, n))
        minutes = _MINUTES[obs_id][n] if kwargs['constraints']['has_resources'] else 0
        return SimpleNamespace(visibility_mask=[minutes > 0] * 4, remaining_minutes=minutes)

    with patch.object(collector_mod, 'site_shim', lambda s: s), \
            patch.object(collector_mod, 'target_shim', lambda b: b), \
            patch.object(collector_mod, 'calculate_night_events_for_night', night_events), \
            patch.object(collector_mod, 'calculate_stage1', stage1), \
            patch.object(collector_mod, 'stage2_constraints', lambda _obs, **kw: kw), \
            patch.object(collector_mod, 'sight_calculate_visibility', visibility), \
            patch.object(collector_mod, 'unpack_array', lambda _b, n: np.zeros(n)), \
            patch.object(collector_mod, 'build_target_info',
                         lambda _s1, rem, _n, start_offset_slots=0: SimpleNamespace(rem_visibility_frac=rem)), \
            patch.object(collector_mod, 'align_to_start', lambda m, _off: m), \
            patch.object(collector_mod, 'resize_to', lambda m, _n: m):
        yield visibility_calls


def _local_collector() -> Collector:
    collector = _collector()
    times = [[SimpleNamespace(to_datetime=lambda _tz, n=n: _START + timedelta(days=n))] * 4 for n in range(3)]
    collector.night_events = {_SITE: SimpleNamespace(times=times)}
    return collector


def test_local_skips_the_remaining_nights_of_observations_not_visible_tonight(local_stages):
    collector = _local_collector()
    tonight, later = _obs('TONIGHT'), _obs('LATER')
    every_night = {NightIndex(n): [tonight, later] for n in range(3)}

    collector._compute_visibility_locally([('p', tonight), ('p', later)], every_night)

    assert local_stages == [('TONIGHT', 0), ('TONIGHT', 1), ('TONIGHT', 2), ('LATER', 0)]
    assert ('target-LATER', later.id) not in collector._target_info
    target_info = collector._target_info[('target-TONIGHT', tonight.id)]
    # Remaining exec time over the visible minutes from each night to the end of the period.
    assert target_info[NightIndex(0)].rem_visibility_frac == pytest.approx(30 / sum(_MINUTES['TONIGHT']))
    assert target_info[NightIndex(2)].rem_visibility_frac == pytest.approx(30 / _MINUTES['TONIGHT'][2])


def test_local_computes_every_night_when_every_night_is_scheduled(local_stages):
    collector = _local_collector()
    collector.num_of_nights = 3
    later = _obs('LATER')

    collector._compute_visibility_locally([('p', later)], {NightIndex(n): [later] for n in range(3)})

    assert local_stages == [('LATER', 0), ('LATER', 1), ('LATER', 2)]
    assert set(collector._target_info[('target-LATER', later.id)]) == {0, 1, 2}
