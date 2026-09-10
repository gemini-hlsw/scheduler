# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import os
import signal
import socket

from lucupy.observatory.abstract import ObservatoryProperties
from lucupy.observatory.gemini.geminiproperties import GeminiProperties

from scheduler.config import config
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
# Configured in config.yaml under `visibility_aggregator`.
_HEARTBEAT_INTERVAL_SECONDS = float(config.visibility_aggregator.heartbeat_interval_seconds)


def _holder() -> str:
    """Identify this run for the coordination row (Heroku dyno id, else host)."""
    return os.environ.get("DYNO") or socket.gethostname()


def _install_signal_handlers() -> None:
    """Cancel the run on SIGTERM/SIGINT so Heroku dyno shutdown releases the
    coordination row cleanly instead of waiting out the staleness window."""
    try:
        loop = asyncio.get_running_loop()
        main_task = asyncio.current_task()
    except RuntimeError:  # pragma: no cover - no running loop
        return

    def _cancel() -> None:
        _logger.warning("Shutdown signal received; stopping aggregation run.")
        if main_task is not None:
            main_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _cancel)
        except (NotImplementedError, RuntimeError):  # pragma: no cover
            pass  # unsupported platform / not the main thread


async def _run() -> int:
    ObservatoryProperties.set_properties(GeminiProperties)
    _install_signal_handlers()
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
            # Atomically claim the aggregator row.
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
                    work,
                    heartbeat=_make_detail_heartbeat()
                )
            _logger.info(f"Aggregation complete: {result}")
        finally:
            # Stop heartbeating and release the row even on cancellation, so the
            # operation process can resume planning immediately. (Already-committed
            # batches persist; a killed run resumes from the gaps next tick.)
            stop_heartbeat.set()
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            try:
                async with session_scope() as release:
                    await coordination.release_aggregator(release)
            except Exception as exc:
                _logger.warning(
                    f"Could not release the coordination row cleanly ({exc}); "
                    f"it will expire after stale_after_seconds."
                )
        return 0
    except asyncio.CancelledError:
        _logger.warning(
            "Aggregation run cancelled by shutdown signal; coordination row "
            "released and progress committed up to the last batch."
        )
        return 0
    except Exception as exc:
        # e.g. a dropped Heroku Postgres connection mid-run. The row is already
        # released by the inner finally; committed batches persist, so the next
        # cron tick resumes from the remaining gaps.
        _logger.error(f"Aggregation run failed: {exc}", exc_info=True)
        return 1
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
