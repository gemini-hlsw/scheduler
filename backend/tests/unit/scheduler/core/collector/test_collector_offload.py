# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""The collector build must not block the event loop.

`build()` runs on every event, so program parsing, the per-night resource filter and the
visibility computation all landed on the loop the ODB subscription depends on. The
visibility path is the worst of them: for nonsidereal targets it reaches JPL Horizons over
blocking HTTP, so its duration is not even bounded by local CPU.

What must NOT move off the loop is the ODB fetch and the Sight DB read: the GPP client is
keyed per event loop and its `client` property calls `asyncio.get_running_loop()`, and the
Sight AsyncSession is bound to the loop that created it.
"""

import asyncio
import inspect
import threading
from unittest.mock import MagicMock, patch

import pytest
from lucupy.minimodel import Site

from scheduler.core.components.collector import Collector
from scheduler.core.programprovider.abstract import ProgramProvider

_SITE = Site.GN


def _collector() -> Collector:
    """A Collector holding only what the offloaded helpers read.

    __post_init__ builds night events and loads resource files; none of that is needed
    here, so it is bypassed.
    """
    collector = Collector.__new__(Collector)
    collector.sites = frozenset({_SITE})
    collector.num_nights_calculated = 1
    # Needed only to construct the provider inside async_load_programs.
    collector.obs_classes = frozenset()
    collector.sources = MagicMock()
    # __new__ skips __init__, so the dataclass default_factory never runs.
    collector._programs = {}
    collector._observations = {}
    collector._observations_per_program = {}
    collector._target_info = {}
    collector._visible_obs_by_night = {}
    return collector


@pytest.mark.asyncio
async def test_program_parsing_runs_off_the_loop():
    collector = _collector()
    ran_on = {}

    def parse(_provider, _data):
        ran_on["thread"] = threading.get_ident()
        return [("p-1", MagicMock())], 0

    collector.night_configurations = MagicMock(return_value={})
    collector._parse_programs = parse
    collector._filter_by_night_configuration = MagicMock(return_value={})
    collector._use_local_visibility = MagicMock(return_value=True)
    collector._compute_visibility_locally = MagicMock()

    await collector.async_load_programs(_DummyProvider, [])

    assert ran_on["thread"] != threading.get_ident(), \
        "parsing on the loop's thread starves the ODB subscription"


@pytest.mark.asyncio
async def test_the_night_config_filter_runs_off_the_loop():
    collector = _collector()
    ran_on = {}

    def filter_obs(_parsed, _nc):
        ran_on["thread"] = threading.get_ident()
        return {}

    collector.night_configurations = MagicMock(return_value={})
    collector._parse_programs = MagicMock(return_value=([("p-1", MagicMock())], 0))
    collector._filter_by_night_configuration = filter_obs
    collector._use_local_visibility = MagicMock(return_value=True)
    collector._compute_visibility_locally = MagicMock()

    await collector.async_load_programs(_DummyProvider, [])

    assert ran_on["thread"] != threading.get_ident()


@pytest.mark.asyncio
async def test_local_visibility_runs_off_the_loop():
    """The heaviest span, and the only one that can also block on external HTTP."""
    collector = _collector()
    ran_on = {}

    def compute(_parsed, _obs):
        ran_on["thread"] = threading.get_ident()

    collector.night_configurations = MagicMock(return_value={})
    collector._parse_programs = MagicMock(return_value=([("p-1", MagicMock())], 0))
    collector._filter_by_night_configuration = MagicMock(return_value={})
    collector._use_local_visibility = MagicMock(return_value=True)
    collector._compute_visibility_locally = compute

    await collector.async_load_programs(_DummyProvider, [])

    assert ran_on["thread"] != threading.get_ident()


@pytest.mark.asyncio
async def test_an_empty_parse_still_raises_on_the_loop():
    # The guard must survive the offload; a build with no observations is a hard failure.
    collector = _collector()
    collector.night_configurations = MagicMock(return_value={})
    collector._parse_programs = MagicMock(return_value=([], 0))

    with pytest.raises(Exception, match="No observations found"):
        await collector.async_load_programs(_DummyProvider, [])


def test_the_sight_fetch_is_not_offloaded():
    """Its AsyncSession is bound to the loop that made it, so it must stay awaited.

    Only the TargetInfo build after it may move to a thread.
    """
    src = inspect.getsource(Collector._async_load_visibility_from_sight)
    assert "await self._fetch_sight_data" in src, "the Sight read must stay on the loop"
    assert "to_thread(self._apply_sight_visibility" in src, \
        "the TargetInfo build is CPU-bound and belongs off the loop"


def test_the_odb_fetch_is_not_offloaded():
    """The GPP client is per-event-loop and raises with no running loop.

    A thread doing asyncio.run would build a second loop and a second client per replan,
    and churn the registry the ODB subscription's own client lives in.
    """
    from scheduler.core.builder import simulationbuilder
    src = inspect.getsource(simulationbuilder.SimulationBuilder.async_build_collector)
    assert "await gpp_program_data" in src
    assert "to_thread" not in src


# A concrete ProgramProvider is required only to pass async_load_programs' type guard --
# every test here stubs _parse_programs, so none of these methods is ever called. Built
# from __abstractmethods__ so a new abstract method does not silently break these tests.
_DummyProvider = type(
    '_DummyProvider',
    (ProgramProvider,),
    {name: (lambda self, *a, **k: None) for name in ProgramProvider.__abstractmethods__},
)
