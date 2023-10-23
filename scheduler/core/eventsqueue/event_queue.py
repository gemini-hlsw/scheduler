# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from collections import deque
from typing import Deque, FrozenSet, Iterable, Optional

from lucupy.minimodel import NightIndex, Site

from scheduler.services import logger_factory
from .events import Blockage, Event, Interruption, ResumeNight

logger = logger_factory.create_logger(__name__)


class EventQueue:
    def __init__(self, night_indices: FrozenSet[NightIndex], sites: FrozenSet[Site]):
        self._events = {site: {night_idx: deque([]) for night_idx in night_indices} for site in sites}
        self._blockage_stack = []

    def add_event(self, site: Site, night_idx: NightIndex, event: Event) -> None:
        match event:
            case Blockage():
                self._blockage_stack.append(event)
            case Interruption():
                site_deque = self.get_night_events(site, night_idx)
                if site_deque is not None:
                    site_deque.append(event)
                else:
                    raise KeyError(f'Could not add event {event} to site {site.name} for night index {night_idx}.')

    def add_events(self, site: Site, events: Iterable[Event], night_idx: NightIndex) -> None:
        for event in events:
            self.add_event(site, night_idx, event)

    def check_blockage(self, resume_event: ResumeNight) -> Blockage:
        if self._blockage_stack and len(self._blockage_stack) == 1:
            b = self._blockage_stack.pop()
            b.ends(resume_event.start)
            return b

        raise RuntimeError('Missing blockage for ResumeNight')

    def get_night_events(self, site: Site, night_idx: NightIndex) -> Optional[Deque[Event]]:
        """
        Returns the deque for the site for the night index if it exists, else None.
        """
        site_deques = self._events.get(site)
        if site_deques is None:
            logger.error(f'Tried to access event queue for inactive site {site.name}.')
            return None
        site_deque = site_deques.get(night_idx)
        if site_deque is None:
            logger.error(f'Tried to access event queue for site {site.name} for inactive night index {night_idx}.')
        return site_deque
