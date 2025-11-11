# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod
from datetime import timedelta
from typing import ClassVar

from gpp_client.api.custom_fields import TargetEnvironmentFields
from pydantic import BaseModel

class MockObservation(BaseModel):
    """
    gpp client should have these base model
    for now we used this until gpp client is hooked up
    In this case the model is similar to the minimodel but NOT the same.
    """
    id: str
    target_environment: TargetEnvironmentFields

class MockObscalcUpdate(BaseModel):
    """gpp client should have these base model
     for now we used this until gpp client is hooked up """

    editType: str
    oldState: str
    newState: str
    observationId: str
    value: MockObservation


class LastPlanMock:
    visits = []

__all__ = [
    'EventHandler',
    'ResourceEventHandler',
    'WeatherEventHandler',
    'ODBEventHandler'
]

class EventHandler(ABC):

    @abstractmethod
    def parse_event(self, raw_event: dict):
        pass
    @abstractmethod
    async def handle(self, event):
        pass


class ResourceEventHandler(EventHandler):

    def parse_event(self, raw_event: dict):
        pass
    async def handle(self, event):
        pass


class WeatherEventHandler(EventHandler):

    def parse_event(self, raw_event: dict):
        pass

    async def handle(self, event):
        pass

class ODBEventHandler(EventHandler):

    WAITING_THRESHOLD: ClassVar[timedelta] = timedelta(minutes=10)

    async def _on_created(self, event: MockObscalcUpdate):
        """
        Handles the logic when an observation was created.
        """
        # If the observation is a ToO we trigger a new plan request
        too = event.value.target_environment.first_science_target(include_deleted=False).opportunity()
        if too is not None:
            # Do a new schedule
            pass
        # Otherwise we discard the event ?

    async def _on_deleted(self, event: MockObscalcUpdate):

        # Retrieve last plan
        last_plan = LastPlanMock() # plandb_client.get_last_plan()

        if event.observationId in last_plan:
            # TODO: If we keep the ObservationID wrapper this would require a modification
            # Do a new schedule
            pass

    async def _on_updated(self, event):

        if event.new_state == 'READY':
            # Calculations ended. Lets retrieve the current sequence
            obs_id = event.observationId
            last_plan = LastPlanMock() # plandb_client.get_last_plan()
            sequence = [] # gpp.client.get_sequence(obs_id)
            last_visit = last_plan.visits[-1]

            if len(last_visit.sequence) != len(sequence):
                print('Observation got modified from later plan, trigger a new schedule')

            for old_atom, new_atom in zip(last_visit.sequence, sequence):
                # TODO: This comparison needs to be done by visit instead fo atom as sequence might change
                # TODO: when the real execution happens.
                # Different plan structure
                if old_atom.status != new_atom.status:
                    # do a new plan
                    pass

                # Visit past the waiting threshold
                if new_atom.end_time - old_atom.end_time > ODBEventHandler.WAITING_THRESHOLD:
                    # do a new plan
                    pass


    def parse_event(self, raw_event: dict):
        event = MockObscalcUpdate.model_validate(raw_event) # Call pydantic model
        return event

    async def handle(self, event: MockObscalcUpdate):

        # Check type of event
        match event.editType:
            case 'created':
               await self._on_created(event)
            case 'updated':
                await self._on_updated(event)
            case 'hard_delete':
                await self._on_deleted(event)
            case _:
                raise NotImplementedError(f'Missing logic for this type of edit {event.editType}')
