# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import os
import socket

from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini.geminiproperties import GeminiProperties

from scheduler.services import logger_factory
from scheduler.services.sight.database.connection import (
    dispose_engine,
    init_db_engine,
    session_scope,
)
from scheduler.services.visibility_aggregator import coordination
from scheduler.services.visibility_aggregator.aggregator import (
    is_night_in_progress,
    run_aggregation,
)

_logger = logger_factory.create_logger(__name__, with_id=False)

# Refresh the coordination heartbeat at least this often, independent of work
# granularity, so the operation process's staleness check stays accurate.
_HEARTBEAT_INTERVAL_SECONDS = 60.0


def _holder() -> str:
    """Identify this run for the coordination row (Heroku dyno id, else host)."""
    return os.environ.get("DYNO") or socket.gethostname()


async def _run() -> int:
    ObservatoryProperties.set_properties(GeminiProperties)
    await init_db_engine()
    try:
        # Never run while a night is being executed.
        if is_night_in_progress():
            _logger.info("A night is in progress; skipping aggregation cycle.")
            return 0

        async with session_scope() as ctrl:
            if await coordination.is_plan_in_progress(ctrl):
                _logger.info(
                    "Operation process is computing a plan; skipping cycle."
                )
                return 0
            # (2) Atomically claim the aggregator row.
            acquired = await coordination.acquire_aggregator(ctrl, _holder())
        if not acquired:
            _logger.info(
                "Another aggregator run holds the lock (or it is fresh); "
                "skipping cycle."
            )
            return 0

        stop_heartbeat = asyncio.Event()
        heartbeat_task = asyncio.create_task(_heartbeat_loop(stop_heartbeat))
        try:
            async with session_scope() as work:
                result = await run_aggregation(
                    work, heartbeat=_make_detail_heartbeat()
                )
            _logger.info(f"Aggregation complete: {result}")
        finally:
            stop_heartbeat.set()
            await heartbeat_task
            async with session_scope() as release:
                await coordination.release_aggregator(release)
        return 0
    finally:
        await dispose_engine()


def _make_detail_heartbeat():
    """Heartbeat callback that records progress detail on its own session."""
    async def _heartbeat(detail: dict) -> None:
        async with session_scope() as session:
            await coordination.heartbeat_aggregator(session, detail)
    return _heartbeat


async def _heartbeat_loop(stop: asyncio.Event) -> None:
    """Keep the coordination row fresh on a fixed cadence until stopped."""
    while not stop.is_set():
        try:
            async with session_scope() as session:
                await coordination.heartbeat_aggregator(session)
        except Exception as exc:  # pragma: no cover - heartbeat must not crash run
            _logger.warning(f"Heartbeat failed: {exc}")
        try:
            await asyncio.wait_for(stop.wait(), timeout=_HEARTBEAT_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass


def main() -> None:
    """Cron entrypoint for the visibility aggregator.

    Designed to be run by Heroku Scheduler as a one-off dyno:

    $ cd /home && uv run python -m scheduler.services.visibility_aggregator.runner

    Each invocation:
      1. exits immediately if a night is in progress (dark time at GN/GS) or the
         operation process is mid plan-computation — so we never run while nights
         are being executed;
      2. atomically acquires the ``visibility_aggregator`` coordination row (so two
         overlapping ticks can't both run, and so the operation process blocks plan
         creation while we work);
      3. brings the Sight DB up to date for the current semester (sidereal only),
         heartbeating throughout;
      4. always releases the row on the way out.
    """
    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
