# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""A loop stall must be measurable.

Blocking work on the loop is otherwise invisible until something downstream times out --
the symptom we actually saw was the ODB websocket giving up after 10s, which reads like a
network fault and is not one. This is the regression alarm for the offload.
"""

import asyncio
import time

import pytest

from scheduler.services.loop_monitor import LoopMonitor


@pytest.mark.asyncio
async def test_a_healthy_loop_reports_no_stall():
    monitor = LoopMonitor(interval=0.01, warn_after=0.5)
    monitor.start()
    await asyncio.sleep(0.1)
    await monitor.stop()

    assert monitor.worst_stall < 0.5, \
        f"an idle loop should not look stalled (saw {monitor.worst_stall:.3f}s)"


@pytest.mark.asyncio
async def test_a_blocking_call_is_measured():
    monitor = LoopMonitor(interval=0.01, warn_after=10.0)
    monitor.start()
    await asyncio.sleep(0.02)

    # Exactly the thing the offload removes: sync CPU on the loop.
    time.sleep(0.3)
    await asyncio.sleep(0.02)
    await monitor.stop()

    assert monitor.worst_stall >= 0.2, \
        f"a 0.3s block must be visible (saw {monitor.worst_stall:.3f}s)"


@pytest.mark.asyncio
async def test_a_stall_over_the_threshold_warns():
    monitor = LoopMonitor(interval=0.01, warn_after=0.1)
    monitor.start()
    await asyncio.sleep(0.02)
    time.sleep(0.25)
    await asyncio.sleep(0.02)
    await monitor.stop()

    assert monitor.worst_stall >= 0.1


@pytest.mark.asyncio
async def test_start_is_idempotent():
    # NightMonitor.start may be reached more than once; a second sampler would double-log.
    monitor = LoopMonitor(interval=0.01)
    monitor.start()
    first = monitor._task
    monitor.start()

    assert monitor._task is first
    await monitor.stop()


@pytest.mark.asyncio
async def test_stop_without_start_is_safe():
    await LoopMonitor().stop()
