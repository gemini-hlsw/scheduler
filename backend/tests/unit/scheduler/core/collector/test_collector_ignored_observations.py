# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Ignored observations get no visibility, so the plan cannot place them.

They are dropped before the visibility load rather than zeroed after it, so
ignoring an observation makes a build cheaper, never slower. They stay in the
Collector's observation table: the Night Monitor still has to recognise them.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from lucupy.minimodel import ObservationID, Site

from scheduler.core.components.collector import Collector
from scheduler.core.programprovider.abstract import ProgramProvider

_SITE = Site.GN
_KEPT = "G-2026B-0001-Q-0001"
_IGNORED = "G-2026B-0001-Q-0002"


# Only passes the Collector's provider type check: _parse_programs is stubbed.
_DummyProvider = type(
    '_DummyProvider',
    (ProgramProvider,),
    {name: (lambda self, *a, **k: None) for name in ProgramProvider.__abstractmethods__},
)


def _observation(label: str) -> MagicMock:
    observation = MagicMock()
    observation.id = ObservationID(label)
    return observation


def _collector(local: bool) -> Collector:
    """A Collector holding only what async_load_programs reads."""
    collector = Collector.__new__(Collector)
    collector.sites = frozenset({_SITE})
    collector.num_nights_calculated = 1
    collector.obs_classes = frozenset()
    collector.sources = MagicMock()
    collector._programs = {}
    collector._observations = {}
    collector._observations_per_program = {}
    collector._target_info = {}
    collector._visible_obs_by_night = {}
    collector.night_configurations = MagicMock(return_value={})
    collector._parse_programs = MagicMock(
        return_value=([("p-1", _observation(_KEPT)), ("p-1", _observation(_IGNORED))], 0)
    )
    collector._use_local_visibility = MagicMock(return_value=local)
    collector._filter_by_night_configuration = MagicMock(return_value={})
    collector._compute_visibility_locally = MagicMock()
    collector._sight_date_args = MagicMock(return_value=(None, None, []))
    collector._fetch_sight_data_for_scheduled_nights = AsyncMock(return_value=({}, {}))
    collector._apply_sight_visibility = MagicMock()
    return collector


def _labels(parsed) -> list[str]:
    return [observation.id.id for _program_id, observation in parsed]


@pytest.mark.asyncio
async def test_an_ignored_observation_is_left_out_of_local_visibility():
    collector = _collector(local=True)
    collector.ignored_observation_ids = frozenset({_IGNORED})

    await collector.async_load_programs(_DummyProvider, [])

    assert _labels(collector._filter_by_night_configuration.call_args.args[0]) == [_KEPT]
    assert _labels(collector._compute_visibility_locally.call_args.args[0]) == [_KEPT]


@pytest.mark.asyncio
async def test_an_ignored_observation_is_left_out_of_the_sight_queries():
    collector = _collector(local=False)
    collector.ignored_observation_ids = frozenset({_IGNORED})

    await collector.async_load_programs(_DummyProvider, [])

    assert _labels(collector._fetch_sight_data_for_scheduled_nights.call_args.args[0]) == [_KEPT]


@pytest.mark.asyncio
async def test_an_ignored_observation_stays_in_the_collector():
    collector = _collector(local=True)
    collector.ignored_observation_ids = frozenset({_IGNORED})

    await collector.async_load_programs(_DummyProvider, [])

    assert {observation_id.id for observation_id in collector._observations} == {_KEPT, _IGNORED}
    assert collector.get_target_info(ObservationID(_IGNORED)) is None


@pytest.mark.asyncio
async def test_nothing_is_dropped_when_nothing_is_ignored():
    collector = _collector(local=True)

    await collector.async_load_programs(_DummyProvider, [])

    assert _labels(collector._compute_visibility_locally.call_args.args[0]) == [_KEPT, _IGNORED]


@pytest.mark.asyncio
async def test_a_build_with_every_observation_ignored_still_completes():
    collector = _collector(local=True)
    collector.ignored_observation_ids = frozenset({_KEPT, _IGNORED})

    await collector.async_load_programs(_DummyProvider, [])

    assert _labels(collector._compute_visibility_locally.call_args.args[0]) == []
    assert len(collector._observations) == 2
