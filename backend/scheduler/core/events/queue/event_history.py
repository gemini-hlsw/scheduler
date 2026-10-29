# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, UTC
from typing import ClassVar, Iterator, Optional

from lucupy.minimodel import Site

from .events import Event

__all__ = ['ReceivedEvent', 'EventHistory']


@dataclass(frozen=True)
class ReceivedEvent:
    """
    An event the Engine took off the scheduler queue, as the UI shows it.

    Attributes:
        id (str): The event's UUID.
        received_at (datetime): When the Engine took it, in UTC.
        event_type (str): The event class, e.g. WeatherChangeEvent.
        sites (tuple[Site, ...]): The sites it applies to; empty when it applies to none in particular.
        event_time (Optional[datetime]): The time carried by the event, if any.
        description (str): The event's human-readable description.
        triggered_plan (bool): Whether the Engine computed a new plan for it.
    """

    id: str
    received_at: datetime
    event_type: str
    sites: tuple[Site, ...]
    event_time: Optional[datetime]
    description: str
    triggered_plan: bool

    @staticmethod
    def from_event(event: Event, triggered_plan: bool) -> 'ReceivedEvent':
        # Some events are raised for every site at once (ALL_SITES), some for none.
        if event.site is None:
            sites = ()
        elif isinstance(event.site, (frozenset, set)):
            sites = tuple(sorted(event.site, key=lambda s: s.name))
        else:
            sites = (event.site,)
        return ReceivedEvent(
            id=str(event.id),
            received_at=datetime.now(UTC),
            event_type=type(event).__name__,
            sites=sites,
            event_time=event.time,
            description=event.description,
            triggered_plan=triggered_plan,
        )


class EventHistory:
    """
    The most recent events the Engine received, and the subscribers to be told about new ones.

    Kept in memory and bounded: it is there so an operator can see what the Scheduler is reacting
    to, not as an audit log.
    """

    MAX_EVENTS: ClassVar[int] = 200

    def __init__(self, max_events: int = MAX_EVENTS):
        self._events: deque[ReceivedEvent] = deque(maxlen=max_events)
        self._subscribers: set[asyncio.Queue] = set()

    def record(self, event: Event, triggered_plan: bool) -> ReceivedEvent:
        received = ReceivedEvent.from_event(event, triggered_plan)
        self._events.append(received)
        for queue in self._subscribers:
            queue.put_nowait(received)
        return received

    def snapshot(self) -> list[ReceivedEvent]:
        """The recorded events, oldest first."""
        return list(self._events)

    @contextmanager
    def subscribe(self) -> Iterator[asyncio.Queue]:
        """
        A queue that gets every event recorded from now on, until the context exits.

        Subscribe before reading the snapshot, so an event recorded in between is not lost.
        """
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)
