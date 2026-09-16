# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
The weather handler's one job beyond parsing: stamp the event with the clock the plan runs on.

The Engine turns ``event.time`` into a timeslot offset from the night's twilight, and the UI
marks a visit executed when it ends before ``event.time``. A wall-clock stamp against a
simulated night therefore both empties the plan and paints every row as already done.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lucupy.minimodel import CloudCover, ImageQuality, Site

from scheduler.engine.params import BuildParameters
from scheduler.night_monitor.event_handlers.weather_event_handler import WeatherEventHandler

_BASE_MODULE = "scheduler.night_monitor.event_handlers.event_handler"

_RAW = {
    "weatherUpdates": {
        "site": "GN",
        "imageQuality": ImageQuality.IQ70.value,
        "cloudCover": CloudCover.CC50.value,
        "windDirection": 90.0,
        "windSpeed": 5.0,
    }
}


def _handler(night_start=None):
    store = MagicMock()
    store.night_start = AsyncMock(return_value=night_start)
    return WeatherEventHandler(scheduler_queue=AsyncMock(), nightly_timeline_store=store)


def _build_params_store(params):
    store = MagicMock()
    store.get = AsyncMock(return_value=params)
    return patch(f"{_BASE_MODULE}.build_params_store", store)


def _queued_event(handler):
    handler.scheduler_queue.add_schedule_event.assert_awaited_once()
    return handler.scheduler_queue.add_schedule_event.await_args.args[0]


@pytest.mark.asyncio
async def test_weather_change_carries_the_simulated_clock():
    handler = _handler()
    simulated_now = datetime(2026, 8, 26, 6, 0, tzinfo=UTC)
    params = BuildParameters(simulated_now=simulated_now)

    with _build_params_store(params):
        await handler.handle("weather_change", _RAW)

    queued = _queued_event(handler)
    # The clock advances with the real one from the moment the params were set, so this is a
    # window rather than an exact instant.
    assert simulated_now <= queued.time < simulated_now + timedelta(minutes=1)
    assert queued.site is Site.GN
    assert queued.variant_change.cc is CloudCover.CC50


@pytest.mark.asyncio
async def test_weather_change_anchors_on_the_nights_twilight_without_a_simulated_now():
    night_start = datetime(2026, 8, 26, 5, 32, tzinfo=UTC)
    handler = _handler(night_start=night_start)
    # Customized (a program list is set) but no simulated_now: fall back to twilight.
    params = BuildParameters(program_list=["G-2026A-0500"])

    with _build_params_store(params):
        await handler.handle("weather_change", _RAW)

    assert night_start <= _queued_event(handler).time < night_start + timedelta(minutes=1)


@pytest.mark.asyncio
async def test_weather_change_uses_the_real_clock_when_not_simulating():
    handler = _handler()
    before = datetime.now(UTC)

    with _build_params_store(BuildParameters()):
        await handler.handle("weather_change", _RAW)

    assert before <= _queued_event(handler).time <= datetime.now(UTC)
