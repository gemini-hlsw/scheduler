# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from uuid import UUID

from sortedcontainers import SortedList
from typing import FrozenSet, Iterable, Optional

from lucupy.minimodel import NightIndex, Site

from scheduler.services import logger_factory
from .events import Blockage, Event, Interruption, ResumeNight

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
    def __init__(self, night_indices: FrozenSet[NightIndex], sites: FrozenSet[Site]):
        self._events = {night_idx: {site: NightEventQueue(night_idx=night_idx, site=site) for site in sites}
                        for night_idx in night_indices}
        self._blockage_stack = []

    def add_event(self, night_idx: NightIndex, site: Site, event: Event) -> None:
        match event:
            case Blockage():
                self._blockage_stack.append(event)
            case Interruption():
                site_events = self.get_night_events(night_idx, site)
                if site_events is not None:
                    site_events.add_event(event)
                else:
                    raise KeyError(f'Could not add event {event} for night index {night_idx} to site {site.name}.')

    def add_events(self, night_idx: NightIndex, site: Site, events: Iterable[Event]) -> None:
        for event in events:
            self.add_event(night_idx, site, event)

    def remove_event(self, night: NightIndex, site: Site, event_id: UUID) -> bool:
        for idx, e in enumerate(self._events[night][site].events):
            if e.event_id == event_id:
                try:
                    self._events[night][site].events.pop(idx)
                except ValueError:
                    return False
                else:
                    return True
        return False





    def check_blockage(self, resume_event: ResumeNight) -> Blockage:
        if self._blockage_stack and len(self._blockage_stack) == 1:
            b = self._blockage_stack.pop()
            b.ends(resume_event.start)
            return b
        raise RuntimeError('Missing blockage for ResumeNight')

    def get_night_events(self, night_idx: NightIndex, site: Site) -> Optional[NightEventQueue]:
        """
        Returns the sorted list for the site for the night index if it exists, else None.
        """
        night_lists = self._events.get(night_idx)
        if night_lists is None:
            logger.error(f'Tried to access event queue for inactive night index {night_idx}.')
            return None
        site_list = night_lists.get(site)
        if site_list is None:
            logger.error(f'Tried to access event queue for night index {night_idx} for inactive site {site.name}.')
        return site_list
