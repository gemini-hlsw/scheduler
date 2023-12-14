# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from datetime import timedelta, datetime

from astropy.time import Time
from sortedcontainers import SortedList
from typing import FrozenSet, Iterable, Optional

from lucupy.minimodel import NightIndex, Site

from scheduler.services import logger_factory
from .events import Event, Interruption

logger = logger_factory.create_logger(__name__)


__all__ = ['EventQueue']


@dataclass
class NightEventQueue:
    night_idx: NightIndex
    site: Site

    # events is a list managed by heapq.
    events: SortedList[Event] = field(init=False, default_factory=lambda: SortedList(key=lambda evt: evt.start))

    def has_more_events(self) -> bool:
        return len(self.events) > 0

    def next_event(self) -> Event:
        return self.events.pop(0)

    def add_event(self, event: Event) -> None:
        self.events.add(event)


class EventQueue:
    def __init__(self, start: Time, end: Time, sites: FrozenSet[Site]):
        curr = start.to_datetime()
        self._events = {}
        night_idx = 0
        while curr <= end.to_datetime():
            self._events[curr] = {site: NightEventQueue(night_idx=NightIndex(night_idx), site=site) for site in sites}
            night_idx += 1
            curr += timedelta(days=1)

        self._blockage_stack = []

    def add_event(self, night_date: datetime, site: Site, event: Event) -> None:
        match event:
            case Interruption():
                site_events = self.get_night_events(night_date, site)
                if site_events is not None:
                    site_events.add_event(event)
                else:
                    raise KeyError(f'Could not add event {event} for date {night_date} to site {site.name}.')

    def get_night_events(self, night_date: datetime, site: Site) -> Optional[NightEventQueue]:
        """
        Returns the sorted list for the site for the night index if it exists, else None.
        """
        night_lists = self._events.get(night_date)
        if night_lists is None:
            logger.error(f'Tried to access event queue for inactive date {night_date}.')
            return None
        site_list = night_lists.get(site)
        if site_list is None:
            logger.error(f'Tried to access event queue for date {night_date} for inactive site {site.name}.')
        return site_list
