# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod
from typing import  Optional, Dict, Tuple, Callable

from gpp_client.api.custom_fields import TargetEnvironmentFields, ConstraintSetFields, \
    CalculatedObservationWorkflowFields, VisitFields

from pydantic import BaseModel

__all__ = [
    'EventHandler',
    'MockObservation',
    'LastPlanMock'
]

class MockObservation(BaseModel):
    """
    gpp client should have these base model
    for now we used this until gpp client is hooked up
    In this case the model is similar to the minimodel but NOT the same.
    """
    id: str
    target_environment: Optional[TargetEnvironmentFields]
    constraint_set: Optional[ConstraintSetFields]
    workflow: Optional[CalculatedObservationWorkflowFields]

    model_config = {
        'arbitrary_types_allowed': True
    }

class MockObservationEdit(BaseModel):
    """gpp client should have these base model
     for now we used this until gpp client is hooked up """

    editType: str
    oldState: str
    newState: str
    observationId: str
    value: MockObservation


class LastPlanMock:
    visits = []

    def get_observation(self, observationId):
        return MockObservation

    def current_visit(self):
        """Pointer to the current visit. Gets updated when a new visit is executed"""
        return VisitFields

    def resources(self):
        return []


class EventHandler(ABC):
    """
    Base class for all event handlers with dispatch pattern support.
    """

    _DISPATCH_MAP: Dict[str, Tuple[callable, callable]]

    def __init__(self):
        self._DISPATCH_MAP = self._build_dispatch_map()

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
            sub_name: The subscription name/event type
            raw_event: Raw event data to parse and handle
        """
        try:
            parser, handler = self._DISPATCH_MAP[sub_name]
        except KeyError:
            raise ValueError(f"Missing subscription for event source: {sub_name}")

        # Parse the raw event
        event = parser(raw_event)

        # Handle the parsed event
        await handler(event)
