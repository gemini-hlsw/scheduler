# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod
from datetime import datetime, UTC
from typing import  Optional, Dict, Tuple, Callable

from gpp_client.generated.custom_fields import TargetEnvironmentFields, ConstraintSetFields, \
    CalculatedObservationWorkflowFields, VisitFields

from lucupy.minimodel import ALL_SITES, Site
from pydantic import BaseModel
from scheduler.core.events.queue.nightly_timeline_store import NightlyTimelineStore
from scheduler.core.events.queue.scheduler_queue_client import SchedulerQueue
from scheduler.engine.params import build_params_store

__all__ = [
    'EventHandler',
    'LastPlanMock'
]


class LastPlanMock:
    visits = []

    def get_observation(self, observationId):
        pass

    def current_visit(self):
        pass

    def resources(self):
        return []


class EventHandler(ABC):
    """
    Base class for all event handlers with dispatch pattern support.
    """

    _DISPATCH_MAP: Dict[str, Tuple[callable, callable]]

    def __init__(self,
                 scheduler_queue: SchedulerQueue,
                 nightly_timeline_store: Optional[NightlyTimelineStore] = None):
        """
        Args:
            scheduler_queue (SchedulerQueue): Use to send new schedule requests to the Engine.
            nightly_timeline_store (Optional[NightlyTimelineStore]): Shared store holding the
                timeline the Engine writes. Read it to check the plan in effect before deciding
                that an event deserves a new schedule.
        """
        self._DISPATCH_MAP = self._build_dispatch_map()
        self.scheduler_queue = scheduler_queue
        self.nightly_timeline_store = nightly_timeline_store

    async def _reference_time(self, site: Optional[Site] = None) -> datetime:
        """
        The current time as far as the plan is concerned.

        Every event that reaches the Engine must be stamped with this rather than the wall clock:
        the Engine turns ``event.time`` into a timeslot offset from the night's twilight, so a real
        timestamp against a simulated night lands thousands of timeslots past the night's end.
        """
        build_params = await build_params_store.get()
        if not build_params.is_customized():
            return datetime.now(UTC)

        anchor = build_params.simulated_now
        if anchor is None:
            anchor = await self._plan_night_start(site)
            if anchor is None:
                # The engine has not recorded a night yet, so there is nothing to anchor to.
                return datetime.now(UTC)

        return anchor + (datetime.now(UTC) - build_params.set_at)

    async def _plan_night_start(self, site: Optional[Site]) -> Optional[datetime]:
        """
        Evening twilight of the night the plan in effect covers.

        ``site`` is None for events raised for every site at once; any site that has a night
        recorded will do there, since both are the same night.
        """
        if self.nightly_timeline_store is None:
            return None
        sites = (site,) if site is not None else tuple(ALL_SITES)
        for candidate in sites:
            night_start = await self.nightly_timeline_store.night_start(candidate)
            if night_start is not None:
                return night_start
        return None

    @abstractmethod
    def _build_dispatch_map(self) -> Dict[str, Tuple[Callable, Callable]]:
        """
        Build and return the dispatch map for this handler.
        Returns a dict mapping subscription names to (parser, handler) tuples.

        Returns:
            Dict[str, Tuple[Callable, Callable]]: Map of sub_name -> (parser, handler)
        """
        pass

    async def handle(self, sub_name: str, raw_event: dict):
        """
        Generic handle method using the dispatch map pattern.

        Args:
            sub_name (str): The subscription name/event type
            raw_event (dict): Raw JSON event data to parse and handle
        """
        try:
            parser, handler = self._DISPATCH_MAP[sub_name]
        except KeyError:
            raise ValueError(f"Missing subscription for event source: {sub_name}")

        # Parse the raw event
        event = parser(raw_event)

        # Handle the parsed event
        await handler(event)
