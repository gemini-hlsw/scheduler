# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import ClassVar, Optional, Dict, Tuple, Callable
from unittest import case

from gpp_client.api.custom_fields import TargetEnvironmentFields, ConstraintSetFields, \
    CalculatedObservationWorkflowFields, VisitFields
from scheduler.clients.scheduler_queue_client import schedule_queue
from pydantic import BaseModel

from scheduler.core.plans import Visit
from scheduler.night_monitor import ODBEventSource


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


__all__ = [
    'EventHandler',
    'ResourceEventHandler',
    'WeatherEventHandler',
    'ODBEventHandler'
]

class EventHandler(ABC):

    @abstractmethod
    async def handle(self, sub_name, event):
        pass


class ResourceEventHandler(EventHandler):

    async def handle(self, sub_name, event):
        pass


class WeatherEventHandler(EventHandler):

    async def handle(self, sub_name, event):
        pass

class ODBEventHandler(EventHandler):
    """
    Handles ODB events. To check the different subscriptions go to ODBEventSource.


    """

    WAITING_THRESHOLD: ClassVar[timedelta] = timedelta(minutes=10)

    _DISPATCH_MAP: Dict[str, Tuple[callable, callable]]

    def __init__(self):
        # Initialize the map, binding the methods to 'self'
        self._DISPATCH_MAP = {
            ODBEventSource.OBSERVATION_EDIT: (
                self.parse_observation_edit_event,
                self._on_observation_edit,
            ),
            # ODBEventSource.VISIT_EXECUTED: (
            #     self.parse_visit_executed_event,
            #     self._on_visit_executed
            # ),
        }

    @staticmethod
    async def _on_created_edit(event: MockObservationEdit):
        """
        A new observation was created. Check if the status is ready for this new observation.
        For ToOs we might want to interrupt so check that status as well.

        Args:
            event (MockObservationEdit): The observation edit type created.
        """
        # If the observation is a ToO we trigger a new plan request
        if event.value.observation.workflow.state == 'READY':
            too = event.value.target_environment.first_science_target(include_deleted=False).opportunity()
            if too is not None:
                # TODO: For now we do nothing until we implement the logic for different types of ToOs.
                pass # Check the type of opportunity
            schedule_queue.add_schedule_event()

    @staticmethod
    async def _on_deleted_edit(event: MockObservationEdit):
        """
        An observation was deleted. Check if is in the current plan to retrieve a new plan.
        Otherwise, we keep the current plan.

        Args:
            event (MockObservationEdit): The observation edit type deleted.
        """
        # Retrieve last plan
        last_plan = LastPlanMock() # plandb_client.get_last_plan()

        if event.observationId in last_plan:
            # TODO: If we keep the ObservationID wrapper this would require a modification
            schedule_queue.add_schedule_event()

    @staticmethod
    async def _on_updated_edit(event: MockObservationEdit):
        """
        An updated edit means the observation was modified.
        Check if the conditions in an observation was changed.

        Args:
            event (MockObservationEdit): The observation edit type updated.
        """
        # Retrieve last plan
        last_plan = LastPlanMock()
        old_observation = last_plan.get_observation(event.observationId)

        old_constraints = old_observation.constraints_set
        new_constraints = event.value.constraint_set

        # TODO: This only work if we keep the pydantic model from gpp-client into the
        # TODO: plan structure (currently we use minimodel Constraints).
        # Constraints changed so we need to trigger a new plan
        if old_constraints != new_constraints:
            schedule_queue.add_schedule_event()

    async def _on_observation_edit(self, event: MockObservationEdit):
        """
        Handles all modifications (edits) to existing observations.

        Args:
            event (MockObservationEdit): The observation edit type.
        """
        # Check type of event
        match event.editType:
            case 'created':
               await self._on_created_edit(event)
            case 'updated':
                await self._on_updated_edit(event)
            case 'hard_delete':
                await self._on_deleted_edit(event)
            case _:
                raise NotImplementedError(f'Missing logic for this type of edit {event.editType}')

    @staticmethod
    async def _on_visit_executed(event):
        """
        Handles when a visit is executed in Navigate and registered in the ODB.
        Allowing the scheduler to check if the last plan is being followed and to update the
        last executed visit.
        """
        # Visit ended. Retrieve last plan
        obs_id = event.observationId
        last_plan = LastPlanMock()  # plandb_client.get_last_plan()
        current_visit_last_plan = last_plan.current_visit()

        new_plan_created = False

        if obs_id != current_visit_last_plan.observation().id:
            # Last executed visit differs from the plan. Do a new plan
            schedule_queue.add_schedule_event()
            return

        new_visit_duration = event.visit.interval().duration().seconds
        current_plan_delta = (
                current_visit_last_plan.interval().duration().seconds
                + ODBEventHandler.WAITING_THRESHOLD.seconds
        )
        # Visit took longer that it should, putting it behind schedule.
        if new_visit_duration > current_plan_delta:
            schedule_queue.add_schedule_event()
            return

        # We are following the plan. Update last executed visit.
        # TODO: PlanDB is not implemented yet, any code put here is just speculation.

    @staticmethod
    def parse_observation_edit_event(raw_event: dict) -> MockObservationEdit:
        event = MockObservationEdit.model_validate(raw_event) # Call pydantic model
        return event

    @staticmethod
    def parse_visit_executed_event(raw_event: dict):
        # No pydantic model exists yet
        pass

    async def handle(self, sub_name: str, raw_event: dict):
        """
        Match the event using a dispatch map for better scalability.
        """
        try:
            # Look up the parser and handler functions
            parser, handler = self._DISPATCH_MAP[sub_name]
        except KeyError:
            # Handle missing subscription
            raise ValueError(f"Missing subscription for event source: {sub_name}")

        # Call the parser with the raw data
        event = parser(raw_event)

        # Call the async handler with the parsed event
        await handler(event)