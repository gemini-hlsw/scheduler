# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
import time
from datetime import timedelta, datetime, UTC
from functools import partial
from typing import ClassVar, Dict, Tuple, Callable, Awaitable, Optional

from scheduler.core.events.queue import (Event, NightlyTimelineStore, ObservationActivationEvent,
                                         OnDemandScheduleEvent)
from scheduler.night_monitor.event_sources import ODBEventSource
from .event_handler import EventHandler, LastPlanMock
from .obscalc_visibility import calculate_and_store_visibility, site_key_from_instrument, sight_visibility_enabled
from gpp_client.generated.enums import ObservationWorkflowState
from gpp_client.generated.scheduler_observations_updates import SchedulerObservationsUpdates, SchedulerObservationsUpdatesObscalcUpdate

from lucupy.minimodel import ALL_SITES, Site, Observation, ObservationID, ObservationStatus

from scheduler.services import logger_factory
from ...core.events.queue.scheduler_queue_client import SchedulerQueue
from ...engine.params import build_params_store

_logger = logger_factory.create_logger(__name__)


class SchedulerIdleTimer:
    """
    A single pending wake-up. Arming replaces whatever was pending, so a timer never runs twice.
    """

    # Grace given to an observation on top of its estimated execution time.
    WAITING_OFFSET: ClassVar[timedelta] = timedelta(minutes=2)

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None

    @staticmethod
    async def _run(delay: float, callback: Callable[[], Awaitable[None]]) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            await callback()
        except asyncio.CancelledError:
            raise
        except Exception:
            _logger.exception("Timer callback failed")

    def set(self, delay: float, callback: Callable[[], Awaitable[None]], with_offset: bool = False) -> None:
        self.cancel()
        if with_offset:
            delay += SchedulerIdleTimer.WAITING_OFFSET.total_seconds()
        self._task = asyncio.create_task(self._run(delay, callback))

    def cancel(self) -> None:
        """Disarm. Safe to call when nothing is pending."""
        task, self._task = self._task, None
        # A callback that rearms timers runs inside this very task; cancelling it there would kill
        # the callback halfway through.
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    @property
    def pending(self) -> bool:
        return self._task is not None and not self._task.done()

class ODBEventHandler(EventHandler):
    """
    Handles ODB events. To check the different subscriptions go to ODBEventSource.
    """

    # How much the Scheduler can wait idle without anything set
    WAITING_THRESHOLD: ClassVar[timedelta] = timedelta(minutes=10)

    _DISPATCH_MAP: Dict[str, Tuple[callable, callable]]


    def __init__(self,
                 scheduler_queue: SchedulerQueue,
                 nightly_timeline_store: Optional[NightlyTimelineStore] = None):
        super().__init__(scheduler_queue, nightly_timeline_store)
        # Per site: the sites execute independently, so a GN observation running must not keep
        # GS from noticing that it has gone quiet.
        self.observation_execution_timer: Dict[Site, SchedulerIdleTimer] = {
            site: SchedulerIdleTimer() for site in ALL_SITES
        }
        self.idle_timer: Dict[Site, SchedulerIdleTimer] = {
            site: SchedulerIdleTimer() for site in ALL_SITES
        }

    @staticmethod
    def _sites_of(event: Event) -> Tuple[Site, ...]:
        """The sites an event applies to. Some are raised for every site at once (ALL_SITES)."""
        return tuple(event.site) if isinstance(event.site, (frozenset, set)) else (event.site,)

    async def _request_new_plan(self, event: Event) -> None:
        """
        Ask the Engine for a new plan.

        Every trigger in this handler goes through here so the pending wait is always dropped
        first: once a new plan is requested, the observation we were waiting on is no longer the
        one the plan expects, and any idle countdown is measuring against a plan about to be
        replaced. The idle watch is then rearmed, so a site that goes quiet after this event
        still gets picked up.
        """
        sites = self._sites_of(event)
        for site in sites:
            self.observation_execution_timer[site].cancel()
            self.idle_timer[site].cancel()

        await self.scheduler_queue.add_schedule_event(event)

        for site in sites:
            self._arm_idle_timer(site)

    def _arm_idle_timer(self, site: Site) -> None:
        """
        Watch a site that is not expected to be executing anything.

        If WAITING_THRESHOLD passes with no ODB activity, the plan in effect is stale enough to
        be worth recomputing (conditions and the clock have moved on), so request a new one.
        """
        self.observation_execution_timer[site].cancel()
        self.idle_timer[site].set(
            ODBEventHandler.WAITING_THRESHOLD.total_seconds(),
            partial(self._on_idle_timeout, site),
        )

    def _arm_execution_timer(self, site: Site, observation: Observation) -> None:
        """
        The plan is being followed, so wait the observation out instead of replanning now.

        The timer runs for the estimated execution time plus WAITING_OFFSET. If the ODB has not
        reported the observation finished by then, we are behind schedule and replan.
        """
        self.idle_timer[site].cancel()
        self.observation_execution_timer[site].set(
            observation.exec_time().total_seconds(),
            partial(self._on_execution_timeout, site, observation.id),
            with_offset=True,
        )

    async def _on_idle_timeout(self, site: Site) -> None:
        """No ODB activity for WAITING_THRESHOLD: recompute so the plan matches the current time."""
        _logger.info(f'{site.name} idle for {ODBEventHandler.WAITING_THRESHOLD}. Requesting a new plan.')
        await self._request_new_plan(
            OnDemandScheduleEvent(
                site=site,
                time=datetime.now(UTC),
                description=f'Scheduler idle at {site.name} for {ODBEventHandler.WAITING_THRESHOLD}. Recalculate...'
            )
        )

    async def _on_execution_timeout(self, site: Site, obs_id: ObservationID) -> None:
        """The observation we were waiting on overran its estimate: the plan is behind, replan."""
        _logger.info(f'Observation {obs_id.id} overran its estimated execution time. Requesting a new plan.')
        await self._request_new_plan(
            ObservationActivationEvent(
                site=site,
                observation_id=obs_id,
                time=datetime.now(UTC),
                description=f'Observation {obs_id.id} did not complete within its estimated execution time. '
                            f'Recalculate...'
            )
        )

    def _build_dispatch_map(self) -> Dict[str, Tuple[Callable, Callable]]:
        return {
            ODBEventSource.OBSERVATION_EDIT: (
                self.parse_observation_edit_event,
                self._on_observation_edit,
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
            await self._request_new_plan(
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
        # Visits stores ObservationIDs, no internal ids
        deleted_obs = event.value.reference.label

        # TODO: OR GROUP support. how do we know an observation belongs two too sites.
        # If is correctly added then it should appear in the other site

        for site in ALL_SITES:
            # Retrieve last plan observations IDs
            obs_ids = await self.nightly_timeline_store.planned_observation_ids(site)
            if deleted_obs in obs_ids:
                await self._request_new_plan(
                    ObservationActivationEvent(
                        site=site,
                        observation_id=event.value.id,
                        time=datetime.now(UTC),
                        description=f'Observation {deleted_obs} deleted from plan'
                    )
                )
                break # to not trigger two events

    async def _on_updated_edit(self, event: SchedulerObservationsUpdatesObscalcUpdate):
        """
        An updated edit is most of the workflow in our current setup
        Check if the conditions in an observation was changed.

        Args:
            event (SchedulerObservationsUpdatesObscalcUpdate): The observation edit type updated.
        """
        instrument = event.value.instrument if event.value.instrument else None
        if instrument is None:
            _logger.warning("Observation without instruments can't be selected!")
            return
        site_key = site_key_from_instrument(instrument)
        site = Site.GN if site_key in 'GN' else Site.GS

        # Time control variables to check when the event happened
        build_params = await build_params_store.get()
        now = datetime.now(UTC)
        when = datetime.combine(build_params.visibility_start.date(), now.time(),
                                tzinfo=UTC) if build_params.visibility_start else now


        updated_obs = event.value
        label = updated_obs.reference.label # lucupy.ObservationID

        last_plan = await self.nightly_timeline_store.last_plan(site)
        if last_plan is None:
            # Nothing has been scheduled for this site yet, so there is no plan to compare against
            # and no timer worth arming.
            _logger.info(f'No plan in effect for {site.name}; skipping update for {label}.')
            return

        last_obs = last_plan.find(ObservationID(label))
        is_in_the_plan = last_obs is not None

        # The observation the plan wants to be running right now, which is not necessarily the one
        # the ODB just told us about.
        expected_obs = None
        for visit in last_plan.visits:
            visit_end = visit.start_time + visit.time_slots * last_plan.time_slot_length
            if visit.start_time <= when < visit_end:
                expected_obs = visit.observation
                break

        if updated_obs.workflow.value.state == 'ONGOING':
            if is_in_the_plan:
                # It was part of the plan
                if expected_obs is None or expected_obs.id.id != label:
                    # We are not following the order of the plan
                    expecting = expected_obs.id.id if expected_obs else 'nothing'
                    await self._request_new_plan(
                        ObservationActivationEvent(
                            site=site,
                            observation_id=updated_obs.id,
                            time=now,
                            description=f'Plan was expecting {expecting} but got {label}. Recalculate...'
                        )
                    )
                else:
                    if last_obs.status is ObservationStatus.READY:
                        # READY -> ONGOING
                        # The plan is being followed, so do not trigger a new plan. Wait out the observation
                        # instead: if it has not completed by the time it should have, the timer
                        # fires and we plan then.
                        self._arm_execution_timer(site, last_obs)

                    # ONGOING -> ONGOING?

            else:
                if expected_obs:
                    msg = f'Expected observation {expected_obs.id.id} in the plan'
                else:
                    msg = 'The plan was not expecting any execution at this time.'

                await self._request_new_plan(
                    ObservationActivationEvent(
                        site=site,
                        observation_id=updated_obs.id,
                        time=datetime.now(UTC),
                        description=f'Observation {label} is currently being executed. {msg}'
                    )
                )
        if updated_obs.workflow.value.state == 'READY':

            if is_in_the_plan:
                # READY -> READY
                # Observation must have changed in some way. Check constraints.
                # We are not saving current conditions yet
                if last_obs.status is ObservationStatus.READY:
                    await self._request_new_plan(
                        ObservationActivationEvent(
                            site=site,
                            observation_id=updated_obs.id,
                            time=datetime.now(UTC),
                            description=f'Observation {label} was modified.'
                        )
                    )
            else:
                # New observation was added to the ODB, we do visibility and trigger a new plan
                if sight_visibility_enabled():
                    # Gated visibility calculation, if strategy set to ``local`` this process is skipped
                    # entirely as the local calculation would be stored when the query is done in the Collector side.

                    t0 = time.perf_counter()
                    label = updated_obs.reference.label
                    if site_key is None or label is None:
                        _logger.warning(
                            f'Skipping visibility for observation {label}: could not resolve '
                            f'site/label (instrument={instrument}, label={label}).'
                        )
                    else:
                        _logger.info(
                            f'Resolved observation {label} -> {label} ({site_key}, {instrument}) '
                            f'in {time.perf_counter() - t0:.2f}s.'
                        )
                        await calculate_and_store_visibility(updated_obs, observation_id=label, site_key=site_key)

                await self._request_new_plan(
                    ObservationActivationEvent(
                        site=site,
                        observation_id=event.value.id,
                        time=datetime.now(UTC),
                        description=f'New observation in the ODB: {updated_obs.reference.label}'
                    )
                )

        if updated_obs.workflow.value.state == 'COMPLETED':
            if is_in_the_plan:
                # The observation we were waiting on finished on time, so drop the execution wait.
                # Nothing is executing now: if the next observation does not start within
                # WAITING_THRESHOLD, the idle timer replans.
                _logger.info(f'Observation {label} completed as planned. Watching {site.name} for idle time.')
                self._arm_idle_timer(site)


    async def _on_observation_edit(self, event: SchedulerObservationsUpdatesObscalcUpdate):
        """
        Handles all modifications (edits) to existing observations.

        Args:
            event (SchedulerObservationsUpdatesObscalcUpdate): The observation edit type.
            scheduler_queue (SchedulerQueue): Use to send new schedule request to the Engine.
        """
        # Check type of event

        _logger.info(
            f'Received ObservationEditEvent:'
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


    @staticmethod
    def parse_observation_edit_event(raw_event: SchedulerObservationsUpdates) -> SchedulerObservationsUpdatesObscalcUpdate:
        return raw_event.obscalc_update
