# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import AsyncIterator, Dict, FrozenSet, List, Optional

from lucupy.minimodel import NightIndex, ObservationID, Site

from scheduler.core.plans import Plan

from .nightchanges import NightlyTimeline, TimelineEntry

__all__ = [
    'NightlyTimelineStore'
]

class NightlyTimelineStore:
    """
    Async-safe owner of the mutable NightlyTimeline shared by the engine and the night monitor.

    The engine writes through :meth:`mutate`; the night monitor handlers read the last plan to
    decide whether an event deserves a new schedule. Only the returned Plan is copied, so a
    reader can never mutate engine state, and no full-timeline copy is ever made.

    The lock is an asyncio.Lock, which only serializes within a single event loop. That holds
    today because SchedulerProcess starts the engine and the night monitor as tasks on the same
    loop. If the engine ever moves onto its own loop in a worker thread, this needs to become
    thread-safe.
    """

    def __init__(self, timeline: Optional[NightlyTimeline] = None) -> None:
        self._timeline = timeline if timeline is not None else NightlyTimeline()
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def mutate(self) -> AsyncIterator[NightlyTimeline]:
        """
        Exclusive access to the live timeline. Writers only.

        Do not hold this across network I/O: readers block for as long as the block runs, and
        the lock is not reentrant, so nesting two ``mutate`` blocks deadlocks.
        """
        async with self._lock:
            yield self._timeline

    async def reset(self) -> None:
        """
        Drop everything recorded so far. Called when the night ends, so the next night does not
        read the previous night's plan as the one in effect.
        """
        async with self._lock:
            self._timeline = NightlyTimeline()

    async def has_plan(self, site: Site) -> bool:
        """Whether a plan has been generated for the site on the current night."""
        async with self._lock:
            return self._last_entry_with_plan(site) is not None

    async def last_plan(self, site: Site) -> Optional[Plan]:
        """
        The plan currently in effect for the site: the last stitched entry that carries one.

        Returns a copy, so the caller can inspect or modify it freely. None when no plan has
        been generated yet for the site.
        """
        async with self._lock:
            entry = self._last_entry_with_plan(site)
            return deepcopy(entry.plan_generated) if entry is not None else None

    async def planned_observation_ids(self, site: Site) -> FrozenSet[ObservationID]:
        """
        The observations in the plan currently in effect for the site.
        """
        async with self._lock:
            entry = self._last_entry_with_plan(site)
            if entry is None:
                return frozenset()
            return frozenset(visit.observation.id for visit in entry.plan_generated.visits)

    def _last_entry_with_plan(self, site: Site) -> Optional[TimelineEntry]:
        """
        The last entry carrying a plan, from the stitched timeline when it has one for the site.

        Events that do not produce a plan (e.g. a fault) are recorded with plan_generated=None,
        so the last entry is not necessarily the last plan. Must be called under the lock.
        """
        entries = (self._entries(self._timeline.stitched_timeline, site)
                   or self._entries(self._timeline.timeline, site))
        for entry in reversed(entries):
            if entry.plan_generated is not None:
                return entry
        return None

    @staticmethod
    def _entries(timeline: Dict[NightIndex, Dict[Site, List[TimelineEntry]]],
                 site: Site) -> List[TimelineEntry]:
        # RT only sgh
        return timeline.get(NightIndex(0), {}).get(site, [])
