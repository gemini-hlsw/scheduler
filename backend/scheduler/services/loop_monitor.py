# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from time import perf_counter
from typing import Optional

from scheduler.services import logger_factory

__all__ = ['LoopMonitor']

_logger = logger_factory.create_logger(__name__)


class LoopMonitor:
    """
    Measures how late the loop wakes a task that asked to sleep a fixed interval.

    The lateness *is* the stall: a healthy loop overshoots `interval` by well under a
    millisecond, so anything above `warn_after` is time some coroutine spent not yielding.
    """

    def __init__(self,
                 interval: float = 0.1,
                 warn_after: float = 1.0) -> None:
        """
        Args:
            interval: seconds between samples. Small enough to catch a stall, large enough
                to be free.
            warn_after: log at WARNING once a single stall exceeds this many seconds.
        """
        self._interval = interval
        self._warn_after = warn_after
        self._task: Optional[asyncio.Task] = None
        self._worst = 0.0

    @property
    def worst_stall(self) -> float:
        """The largest stall seen so far, in seconds. Handy in tests and at shutdown."""
        return self._worst

    async def _run(self) -> None:
        while True:
            before = perf_counter()
            await asyncio.sleep(self._interval)
            # Anything beyond the interval is time the loop could not give us.
            lateness = (perf_counter() - before) - self._interval
            if lateness > self._worst:
                self._worst = lateness
            if lateness >= self._warn_after:
                _logger.warning(
                    f"Event loop stalled {lateness:.1f}s. Something on the loop is doing "
                    f"blocking work; the ODB subscription may have been dropped."
                )

    def start(self) -> None:
        """Begin sampling. Safe to call twice; the second call is a no-op."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())
        _logger.info(
            f"Loop monitor started (sampling every {self._interval * 1000:.0f}ms, "
            f"warning above {self._warn_after:.1f}s)."
        )

    async def stop(self) -> None:
        """Stop sampling and report the worst stall seen."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        _logger.info(f"Loop monitor stopped. Worst stall seen: {self._worst:.2f}s.")
