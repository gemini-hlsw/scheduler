# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""RT-25: EngineRT offloads build/compute to an executor; the event loop must
stay responsive while the worker crunches."""

import asyncio
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from lucupy.minimodel import CloudCover, ImageQuality, Site

from scheduler.engine.engineRT import EngineRT

SITE = Site.GN

GN_WEATHER_STATE = [{
    "site": "GN",
    "imageQuality": ImageQuality.IQ70.value,
    "cloudCover": CloudCover.CC50.value,
    "windDirection": 330.0,
    "windSpeed": 5.0,
}]


def make_engine(executor=None):
    params = MagicMock()
    params.sites = frozenset([SITE])
    weather_source = MagicMock()
    weather_source.get_current_state = AsyncMock(return_value=GN_WEATHER_STATE)
    return EngineRT(params, MagicMock(), "rt25-test",
                    weather_source=weather_source, executor=executor)


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_while_worker_computes(rt_event_factory, mock_coordination_idle):
    """A concurrent coroutine keeps making progress while build+compute block
    inside the (stubbed) worker."""
    fake_plans = MagicMock()

    def slow_build(params, build_params):
        time.sleep(0.15)
        return True

    def slow_compute(payload):
        time.sleep(0.15)
        return fake_plans

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    with ThreadPoolExecutor(max_workers=1) as pool:
        engine = make_engine(executor=pool)
        with patch('scheduler.engine.engineRT.worker_build', slow_build), \
             patch('scheduler.engine.engineRT.worker_compute', slow_compute), \
             patch('scheduler.engine.engineRT.SPlans') as mock_splans:
            ticker_task = asyncio.create_task(ticker())
            await engine.compute_event_plan(rt_event_factory.on_demand())
            ticker_task.cancel()

    assert ticks >= 10, (f"event loop only ticked {ticks} times during ~0.3s of "
                         f"worker computation: the loop was blocked")
    mock_splans.from_computed_plans.assert_called_once_with(fake_plans, engine.params.sites)


@pytest.mark.asyncio
async def test_build_and_compute_are_forwarded_to_worker(rt_event_factory, mock_coordination_idle):
    """The engine sends a build command then a compute payload (event, sites,
    weather variants) to the worker functions, inside the interlock window."""
    fake_plans = MagicMock()
    recorded = {}

    def fake_build(params, build_params):
        recorded['build'] = (params, build_params)
        return True

    def fake_compute(payload):
        recorded['payload'] = payload
        return fake_plans

    event = rt_event_factory.on_demand()

    with ThreadPoolExecutor(max_workers=1) as pool:
        engine = make_engine(executor=pool)
        with patch('scheduler.engine.engineRT.worker_build', fake_build), \
             patch('scheduler.engine.engineRT.worker_compute', fake_compute), \
             patch('scheduler.engine.engineRT.SPlans'):
            await engine.compute_event_plan(event)

    build_params_sent = recorded['build']
    assert build_params_sent[0] is engine.params

    payload = recorded['payload']
    assert payload.event is event
    assert payload.sites == engine.params.sites
    assert payload.variants[SITE].iq == ImageQuality.IQ70
    assert payload.variants[SITE].cc == CloudCover.CC50

    # The interlock bracketed the computation.
    mock_coordination_idle.wait_until_aggregator_idle.assert_awaited_once()
    mock_coordination_idle.signal_plan_in_progress.assert_awaited_once()
    mock_coordination_idle.signal_plan_done.assert_awaited_once()


def test_default_executor_is_a_single_worker_process_pool():
    engine = make_engine()
    executor = engine._get_executor()
    try:
        assert isinstance(executor, ProcessPoolExecutor)
        assert executor._max_workers == 1
        assert engine._get_executor() is executor, "executor must be created once"
    finally:
        engine.shutdown_workers()
    assert engine._executor is None
