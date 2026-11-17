# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import threading
from datetime import datetime, timedelta, UTC
from typing import Any, Dict, Iterable, Optional, Tuple

from gpp_client.generated.enums import ExecutionEventType, SequenceCommand, SlewStage
from lucupy.minimodel import ObservationID, Site

from scheduler.clients.gpp import gpp
from scheduler.core.events.queue.events import OngoingObservation
from scheduler.services import logger_factory

_logger = logger_factory.create_logger(__name__)

__all__ = ['ExecutionTracker']


class ExecutionTracker:
    """
    Knows what each site is executing right now, from the ODB execution events (SLEW and SEQUENCE).

    An observation starts executing with a START_SLEW or, when no slew is needed, a sequence START,
    and stops with a sequence STOP or ABORT. It is expected to end at its start plus the ODB's time
    estimate carried by that event. GN and GS execute at the same time, so each site is tracked on its own.
    """

    def __init__(self):
        self._current: Dict[Site, OngoingObservation] = {}
        self._lock = threading.Lock()

    async def start(self):
        try:
            async for event in gpp.client.scheduler.subscribe_to_executed_events():
                try:
                    self._handle_event(event.execution_event_added.value)
                except Exception:
                    _logger.exception(f'Could not process execution event: {event}')
        finally:
            # Nothing would clear a site once the subscription is gone, and every plan would keep waiting on it.
            _logger.warning('Execution events subscription ended; forgetting what each site is executing.')
            await self.stop()

    async def stop(self):
        """
        Forgets what every site is executing.
        """
        with self._lock:
            self._current.clear()

    def ongoing(self, sites: Iterable[Site], exclude: Optional[ObservationID] = None) -> Dict[Site, OngoingObservation]:
        """
        What each of the sites is executing now.

        Args:
            exclude (Optional[ObservationID]): an observation just taken out of execution (deleted, inactive),
                which will not finish however long it is waited for.
        """
        with self._lock:
            return {
                site: self._current[site]
                for site in sites
                if site in self._current and self._current[site].obs_id != exclude
            }

    def _handle_event(self, value: Any) -> None:
        if value.event_type == ExecutionEventType.SLEW:
            if value.slew_stage == SlewStage.START_SLEW:
                self._start_execution(*self._parse_event(value, value.recorded_time))

        elif value.event_type == ExecutionEventType.SEQUENCE:
            obs_id, site, time, estimate = self._parse_event(value, value.effective_time)
            if value.command == SequenceCommand.START:
                self._start_execution(obs_id, site, time, estimate)
            elif value.command in (SequenceCommand.STOP, SequenceCommand.ABORT):
                self._end_execution(obs_id, site)
            # PAUSE and CONTINUE leave the observation executing.

    @staticmethod
    def _parse_event(value: Any, timestamp: Any) -> Tuple[ObservationID, Site, datetime, Optional[timedelta]]:
        observation = value.observation
        obs_id = ObservationID(observation.reference.label if observation.reference else observation.id)
        time = datetime.fromisoformat(str(timestamp))
        if time.tzinfo is None:
            time = time.replace(tzinfo=UTC)
        digest = observation.execution.digest
        estimate = None
        if digest is not None and digest.value is not None:
            estimate = timedelta(seconds=float(digest.value.estimate.total.total.seconds))
        return obs_id, Site[value.visit.site.value], time, estimate

    def _start_execution(
        self, obs_id: ObservationID, site: Site, time: datetime, estimate: Optional[timedelta]
    ) -> None:
        with self._lock:
            current = self._current.get(site)
            if current is not None:
                # The sequence START that follows a slew, or a re-slew, belongs to the same execution.
                if current.obs_id == obs_id:
                    return
                _logger.warning(f'{current.obs_id.id} at {site.name} never got a STOP; {obs_id.id} replaces it.')
            if estimate is None:
                _logger.warning(f'No time estimate for {obs_id.id}; plans will not wait for it to finish.')
                estimate = timedelta(0)
            self._current[site] = OngoingObservation(obs_id=obs_id, start=time, expected_end=time + estimate)

    def _end_execution(self, obs_id: ObservationID, site: Site) -> None:
        with self._lock:
            current = self._current.get(site)
            if current is None or current.obs_id != obs_id:
                # Usually an execution that started before this tracker was listening.
                _logger.warning(f'End of {obs_id.id} at {site.name} has no matching start, ignoring it.')
                return
            del self._current[site]
