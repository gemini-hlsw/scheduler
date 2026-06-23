# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod
from typing import  Optional, Dict, Tuple, Callable

from gpp_client.generated.custom_fields import TargetEnvironmentFields, ConstraintSetFields, \
    CalculatedObservationWorkflowFields, VisitFields

from pydantic import BaseModel
from scheduler.core.events.queue.scheduler_queue_client import SchedulerQueue

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

    def __init__(self, scheduler_queue: SchedulerQueue):
        self._DISPATCH_MAP = self._build_dispatch_map()
        self.scheduler_queue = scheduler_queue

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
