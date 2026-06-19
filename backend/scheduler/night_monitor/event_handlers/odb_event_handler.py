# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import time
from datetime import timedelta, datetime, UTC
from typing import ClassVar, Dict, Tuple, Callable

from scheduler.clients.gpp import gpp
from scheduler.core.events.queue import ObservationActivationEvent
from scheduler.night_monitor.event_sources import ODBEventSource
from .event_handler import EventHandler, LastPlanMock
from .obscalc_visibility import calculate_and_store_visibility, site_key_from_instrument
from gpp_client.generated.enums import ObservationWorkflowState
from gpp_client.generated.scheduler_observations_updates import SchedulerObservationsUpdates, SchedulerObservationsUpdatesObscalcUpdate

from lucupy.minimodel import ALL_SITES

from scheduler.services import logger_factory

_logger = logger_factory.create_logger(__name__)


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
        A new observation was created. If it is READY, compute and store its
        visibility for the program's active window (so the realtime collector can
        see it before the next semester-wide aggregation run), then trigger a new
        plan request.

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

        value = event.value
        if value is None or value.workflow is None:
            return
        if value.workflow.value.state != ObservationWorkflowState.READY:
            return

        t0 = time.perf_counter()
        try:
            obs = await gpp.client.observation.get_by_id(value.id)
        except Exception as exc:
            _logger.error(f'Could not fetch observation {value.id} to resolve its site: {exc}')
            obs = None

        observation = obs.observation if obs else None
        reference = observation.reference if observation else None
        label = reference.label if reference else None
        instrument = observation.instrument if observation else None
        site_key = site_key_from_instrument(instrument)
        if site_key is None or label is None:
            _logger.warning(
                f'Skipping visibility for observation {value.id}: could not resolve '
                f'site/label (instrument={instrument!r}, label={label!r}).'
            )
        else:
            _logger.info(
                f'Resolved observation {value.id} -> {label} ({site_key}, {instrument!r}) '
                f'in {time.perf_counter() - t0:.2f}s.'
            )
            await calculate_and_store_visibility(value, observation_id=label, site_key=site_key)

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

        _logger.info(
            f'Recieved ObservationEditEvent:'
            f' For observation {event.value.id} -> {event.edit_type}'
            f' Old calculation: {event.old_calculation_state} New calculation: {event.new_calculation_state}'
        )

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
