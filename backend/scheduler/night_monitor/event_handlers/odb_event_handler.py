# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import timedelta, datetime, UTC
from typing import ClassVar, Dict, Tuple, Callable

from scheduler.core.events.queue import ObservationActivationEvent
from scheduler.night_monitor.event_sources import ODBEventSource
from .event_handler import EventHandler, LastPlanMock
from gpp_client.generated.scheduler_observations_updates import SchedulerObservationsUpdates, SchedulerObservationsUpdatesObscalcUpdate

from lucupy.minimodel import ALL_SITES, Site


class ODBEventHandler(EventHandler):
    """
    Handles ODB events. To check the different subscriptions go to ODBEventSource.
    """

    WAITING_THRESHOLD: ClassVar[timedelta] = timedelta(minutes=10)

    _DISPATCH_MAP: Dict[str, Tuple[callable, callable]]

    def _build_dispatch_map(self) -> Dict[str, Tuple[Callable, Callable]]:
        return {
            ODBEventSource.OBSERVATION_EDIT: (
                self.parse_observation_edit_event,
                self._on_observation_edit,
            ),
            ODBEventSource.VISIT_EXECUTED: (
                self.parse_visit_executed_event,
                self._on_visit_executed
            ),
        }

    async def _on_created_edit(self, event: SchedulerObservationsUpdatesObscalcUpdate):
        """
        A new observation was created. Check if the status is ready for this new observation.
        For ToOs we might want to interrupt so check that status as well.

        Args:
            event (SchedulerObservationsUpdatesObscalcUpdate): The observation edit type created.
        """
        # If the observation is a ToO we trigger a new plan request
        if event.value.workflow.value.state == 'READY':
            too = event.value.target_environment.first_science_target(include_deleted=False).opportunity()
            if too is not None:
                # TODO: For now we do nothing until we implement the logic for different types of ToOs.
                pass # Check the type of opportunity

            # TODO create an appropriate event to trigger a new plan, for now we just send
            # an observation activation
            await self.scheduler_queue.add_schedule_event(
                ObservationActivationEvent(
                    site=ALL_SITES,
                    observation_id=event.value.id,
                    time=datetime.now(UTC),
                    description=f'Observation {event.value.id} created from plan: {event.edit_type}'
                )
            )

    async def _on_deleted_edit(self, event: SchedulerObservationsUpdatesObscalcUpdate):
        """
        An observation was deleted. Check if is in the current plan to retrieve a new plan.
        Otherwise, we keep the current plan.

        Args:
            event (MockObservationEdit): The observation edit type deleted.
        """
        # Retrieve last plan
        last_plan = LastPlanMock() # plandb_client.get_last_plan()

        if event.value.id in last_plan:
            # TODO: If we keep the ObservationID wrapper this would require a modification
            # TODO: Create an appropriate event to trigger a new plan
            await self.scheduler_queue.add_schedule_event(
                ObservationActivationEvent(
                    site=ALL_SITES,
                    observation_id=event.value.id,
                    time=datetime.now(UTC),
                    description=f'Observation {event.value.id} deleted from plan: {event.edit_type}'
                )
            )

    async def _on_updated_edit(self, event: SchedulerObservationsUpdatesObscalcUpdate):
        """
        An updated edit means the observation was modified.
        Check if the conditions in an observation was changed.

        Args:
            event (SchedulerObservationsUpdatesObscalcUpdate): The observation edit type updated.
            scheduler_queue (SchedulerQueue): Use to send new schedule request to the Engine.
        """


        # Retrieve last plan
        # last_plan = LastPlanMock()
        # old_observation = last_plan.get_observation(event.value.id)

        # TODO: define when we want to trigger a new plan
        # Recommended to check the workflow state (missing in the gpp-client event for now)
        # Option 1: If the observation is not in the last plan, we trigger a new plan to check if we want to include it in the current schedule.
        # Option 2: If the observation is in the last plan, we check if the constraints changed. If they did, we trigger a new plan to check if we need to update the
        #   TODO get the new constraints in gpp-client

        # For now, we trigger a new plan for any update.
        await self.scheduler_queue.add_schedule_event(
            ObservationActivationEvent(
                site=ALL_SITES,
                observation_id=event.value.id,
                time=datetime.now(UTC),
                description=f'Observation {event.value.id} updated from plan: {event.edit_type}'
            )
        )

    async def _on_observation_edit(self, event: SchedulerObservationsUpdatesObscalcUpdate):
        """
        Handles all modifications (edits) to existing observations.

        Args:
            event (SchedulerObservationsUpdatesObscalcUpdate): The observation edit type.
            scheduler_queue (SchedulerQueue): Use to send new schedule request to the Engine.
        """
        # Check type of event
        match event.edit_type:
            case 'CREATED':
               await self._on_created_edit(event)
            case 'UPDATED':
                await self._on_updated_edit(event)
            case 'HARD_DELETE':
                await self._on_deleted_edit(event)
            case _:
                raise NotImplementedError(f'Missing logic for this type of edit {event.editType}')

    async def _on_visit_executed(self, event):
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
            # TODO: Create an appropriate event to trigger a new plan
            # await self.scheduler_queue.add_schedule_event(VISIT_EXECUTED_EVENT)
            pass

        new_visit_duration = event.visit.interval().duration().seconds
        current_plan_delta = (
                current_visit_last_plan.interval().duration().seconds
                + ODBEventHandler.WAITING_THRESHOLD.seconds
        )
        # Visit took longer that it should, putting it behind schedule.
        if new_visit_duration > current_plan_delta:
            # TODO: Create an appropriate event to trigger a new plan
            # await self.scheduler_queue.add_schedule_event(VISIT_TOO_LONG_EVENT)
            return

        # We are following the plan. Update last executed visit.
        # TODO: PlanDB is not implemented yet, any code put here is just speculation.

    @staticmethod
    def parse_observation_edit_event(raw_event: SchedulerObservationsUpdates) -> SchedulerObservationsUpdatesObscalcUpdate:
        return raw_event.obscalc_update

    @staticmethod
    def parse_visit_executed_event(raw_event: dict):
        # No pydantic model exists yet
        pass
