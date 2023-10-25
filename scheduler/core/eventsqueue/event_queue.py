# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from collections import deque
from typing import Deque, FrozenSet, Iterable, Optional, List

from lucupy.minimodel import NightIndex, Site

from scheduler.services import logger_factory
from .events import Blockage, Event, Interruption, ResumeNight

logger = logger_factory.create_logger(__name__)


class EventQueue:
    def __init__(self, night_indices: FrozenSet[NightIndex], sites: FrozenSet[Site]):
        self._events = {night_idx: {site: deque([]) for site in sites} for night_idx in night_indices}
        self._blockage_stack = {night_idx: {site: [] for site in sites} for night_idx in night_indices}

    def add_event(self, night_idx: NightIndex, site: Site, event: Event) -> None:
        match event:
            case Blockage():
                blockage_stack = self.get_night_stack(night_idx, site)
                if blockage_stack is not None:
                    blockage_stack.append(event)
                else:
                    raise KeyError(f'Could not add event {event} for night index {night_idx} at site {site}')
            case Interruption():
                site_deque = self.get_night_events(night_idx, site)
                if site_deque is not None:
                    site_deque.append(event)
                else:
                    raise KeyError(f'Could not add event {event} for night index {night_idx }to site {site.name}.')

    def add_events(self, night_idx: NightIndex, site: Site, events: Iterable[Event]) -> None:
        for event in events:
            self.add_event(night_idx, site, event)

    def check_blockage(self,
                       night_idx: NightIndex,
                       site: Site,
                       resume_event: ResumeNight) -> Blockage:
        b_stack = self.get_night_stack(night_idx, site)
        if b_stack is None:
            raise KeyError(f'Could not get stack for night index {night_idx} at site {site}')

        if b_stack and len(b_stack) == 1:
            b = b_stack.pop()
            b.ends(resume_event.start)
            return b

        raise RuntimeError('Missing blockage for ResumeNight')

    def get_night_events(self, night_idx: NightIndex, site: Site) -> Optional[Deque[Event]]:
        """
        Returns the deque for the site for the night index if it exists, else None.
        """
        night_deques = self._events.get(night_idx)
        if night_deques is None:
            logger.error(f'Tried to access event queue for inactive night index {night_idx}.')
            return None
        site_deque = night_deques.get(site)
        if site_deque is None:
            logger.error(f'Tried to access event queue for night index {night_idx} for inactive site {site.name}.')
        return site_deque

    def get_night_stack(self, night_idx: NightIndex, site: Site) -> Optional[List[Blockage]]:
        by_night = self._blockage_stack.get(night_idx)
        if by_night:
            stack_by_site = by_night.get(site)
            return stack_by_site if stack_by_site else None
        return None
