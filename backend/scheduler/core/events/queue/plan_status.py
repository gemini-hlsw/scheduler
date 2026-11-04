# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, UTC
from enum import Enum
from typing import Iterator, Optional

from .events import Event, OnDemandScheduleEvent

__all__ = ['PlanState', 'PlanStatus', 'PlanStatusTracker']


class PlanState(Enum):
    """
    What the Engine is doing about a plan.

    Members:
        - IDLE: no plan is being computed.
        - WAITING: a plan was triggered but waits for the visibility aggregator to finish.
        - COMPUTING: a plan is being computed.
    """

    IDLE = 'IDLE'
    WAITING = 'WAITING'
    COMPUTING = 'COMPUTING'


@dataclass(frozen=True)
class PlanStatus:
    """
    Whether the Engine is computing a plan, whoever triggered it, as the UI shows it.

    Attributes:
        state (PlanState): What the Engine is doing.
        since (datetime): When the Engine entered this state, in UTC.
        event_description (Optional[str]): The event the current plan is for; None when IDLE.
        requested_by_user (bool): Whether an operator asked for the current plan from a UI.
        last_finished_at (Optional[datetime]): When the last plan finished, in UTC.
        last_duration_seconds (Optional[float]): How long the last plan took, waiting included.
        last_error (Optional[str]): Why the last plan failed; None when it succeeded.
    """

    state: PlanState
    since: datetime
    event_description: Optional[str] = None
    requested_by_user: bool = False
    last_finished_at: Optional[datetime] = None
    last_duration_seconds: Optional[float] = None
    last_error: Optional[str] = None


class PlanStatusTracker:
    """
    The plan status of one Engine, and the subscribers to be told when it changes.

    The Engine computes one plan at a time, so there is a single current status.
    """

    def __init__(self):
        self._status = PlanStatus(state=PlanState.IDLE, since=datetime.now(UTC))
        self._started_at: Optional[datetime] = None
        self._subscribers: set[asyncio.Queue] = set()

    @property
    def current(self) -> PlanStatus:
        return self._status

    def waiting(self, event: Event) -> None:
        """A plan was triggered for this event, but cannot start yet."""
        self._start(PlanState.WAITING, event)

    def computing(self, event: Event) -> None:
        """The plan for this event is being computed."""
        self._start(PlanState.COMPUTING, event)

    def finished(self, error: Optional[str] = None) -> None:
        """The current plan is done, or failed with the given error."""
        now = datetime.now(UTC)
        started_at = self._started_at or self._status.since
        self._started_at = None
        self._publish(
            PlanStatus(
                state=PlanState.IDLE,
                since=now,
                last_finished_at=now,
                last_duration_seconds=(now - started_at).total_seconds(),
                last_error=error,
            )
        )

    def _start(self, state: PlanState, event: Event) -> None:
        now = datetime.now(UTC)
        # The duration counts from the trigger, so waiting for the aggregator is included.
        if self._status.state is PlanState.IDLE:
            self._started_at = now
        self._publish(
            replace(
                self._status,
                state=state,
                since=now,
                event_description=event.description,
                requested_by_user=isinstance(event, OnDemandScheduleEvent) and event.requested_by_user,
            )
        )

    def _publish(self, status: PlanStatus) -> None:
        self._status = status
        for queue in self._subscribers:
            queue.put_nowait(status)

    @contextmanager
    def subscribe(self) -> Iterator[asyncio.Queue]:
        """
        A queue that gets every status from now on, until the context exits.

        Subscribe before reading `current`, so a change in between is not lost.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)
