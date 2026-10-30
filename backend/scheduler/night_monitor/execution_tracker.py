# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import threading
from dataclasses import dataclass, replace
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Tuple

from gpp_client.generated.enums import ExecutionEventType, SequenceCommand, SlewStage
from lucupy.minimodel import ObservationID, Site

from scheduler.clients.gpp import gpp
from scheduler.services import logger_factory

_logger = logger_factory.create_logger(__name__)

__all__ = ['ExecutionRecord', 'ExecutionTracker']


@dataclass(frozen=True)
class ExecutionRecord:
    """
    One execution of an observation at a site.

    Attributes:
        obs_id (ObservationID): Observation reference label.
        site (Site): Site that executed it.
        start (datetime): Time of the START_SLEW or sequence START that began it.
        end (datetime, optional): Time it ended. None while it is still executing.
        missing_stop (bool): True when another observation started at the site before its STOP arrived.
    """

    obs_id: ObservationID
    site: Site
    start: datetime
    end: Optional[datetime] = None
    missing_stop: bool = False


class ExecutionTracker:
    """
    Tracks observation executions from the ODB execution events (SLEW and SEQUENCE).

    An execution begins with a START_SLEW or, when no slew is needed, a sequence START,
    and ends with a sequence STOP. An ABORT drops it without a record.
    GN and GS execute at the same time, so each site has its own current execution.
    """

    def __init__(self):
        self._executed_observations: List[ExecutionRecord] = []
        self._current_executions: Dict[Site, ExecutionRecord] = {}
        self._lock = threading.Lock()

    async def start(self):
        async for event in gpp.client.scheduler.subscribe_to_executed_events():
            try:
                self._handle_event(event.execution_event_added.value)
            except Exception:
                _logger.exception(f'Could not process execution event: {event}')

    async def stop(self):
        """
        Clears every recorded and current execution.
        """
        with self._lock:
            self._executed_observations.clear()
            self._current_executions.clear()

    def get_executed_observations(
        self, from_: Optional[datetime] = None, to: Optional[datetime] = None
    ) -> List[ExecutionRecord]:
        """
        Executions that overlap [from_, to], oldest first. Current executions are included with no end.

        Args:
            from_ (datetime, optional): UTC start of the range. Open when None.
            to (datetime, optional): UTC end of the range. Open when None.
        """
        with self._lock:
            records = self._executed_observations + list(self._current_executions.values())
        return sorted(
            (
                r
                for r in records
                if (to is None or r.start <= to) and (from_ is None or r.end is None or r.end >= from_)
            ),
            key=lambda r: r.start,
        )

    def is_currently_executed(self, obs_id: ObservationID) -> bool:
        with self._lock:
            return any(r.obs_id == obs_id for r in self._current_executions.values())

    def get_current_execution(self, site: Site) -> Optional[ExecutionRecord]:
        with self._lock:
            return self._current_executions.get(site)

    def _handle_event(self, value: Any) -> None:
        if value.event_type == ExecutionEventType.SLEW:
            if value.slew_stage == SlewStage.START_SLEW:
                self._start_execution(*self._parse_event(value, value.recorded_time))

        elif value.event_type == ExecutionEventType.SEQUENCE:
            obs_id, site, time = self._parse_event(value, value.effective_time)
            if value.command == SequenceCommand.START:
                self._start_execution(obs_id, site, time)
            elif value.command == SequenceCommand.STOP:
                self._stop_execution(obs_id, site, time)
            elif value.command == SequenceCommand.ABORT:
                self._abort_execution(obs_id, site)
            # PAUSE and CONTINUE leave the observation executing.

    @staticmethod
    def _parse_event(value: Any, timestamp: Any) -> Tuple[ObservationID, Site, datetime]:
        observation = value.observation
        obs_id = ObservationID(observation.reference.label if observation.reference else observation.id)
        time = datetime.fromisoformat(str(timestamp))
        if time.tzinfo is None:
            time = time.replace(tzinfo=UTC)
        return obs_id, Site[value.visit.site.value], time

    def _start_execution(self, obs_id: ObservationID, site: Site, time: datetime) -> None:
        with self._lock:
            current = self._current_executions.get(site)
            if current is not None:
                # The sequence START that follows a slew, or a re-slew, belongs to the same execution.
                if current.obs_id == obs_id:
                    return
                _logger.warning(f'{current.obs_id.id} at {site.name} never got a STOP, closing it at {time}.')
                self._executed_observations.append(replace(current, end=time, missing_stop=True))
            self._current_executions[site] = ExecutionRecord(obs_id=obs_id, site=site, start=time)

    def _stop_execution(self, obs_id: ObservationID, site: Site, time: datetime) -> None:
        with self._lock:
            current = self._current_executions.get(site)
            if current is None or current.obs_id != obs_id:
                # Usually an execution that started before this tracker was listening, so its start is unknown.
                _logger.warning(f'STOP for {obs_id.id} at {site.name} has no matching start, ignoring it.')
                return
            self._executed_observations.append(replace(current, end=time))
            del self._current_executions[site]

    def _abort_execution(self, obs_id: ObservationID, site: Site) -> None:
        with self._lock:
            current = self._current_executions.get(site)
            if current is not None and current.obs_id == obs_id:
                del self._current_executions[site]
