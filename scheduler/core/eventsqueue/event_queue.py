# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from collections import deque
from typing import FrozenSet, Iterable

from lucupy.minimodel import NightIndex, Site

from .events import Event, Blockage, ResumeNight


class EventQueue:
    def __init__(self, night_indices: FrozenSet[NightIndex], sites: FrozenSet[Site]):
        self._events = {n_idx: {site: deque([]) for site in sites} for n_idx in night_indices}
        self._blockage_stack = []

    def add_event(self, site: Site, night_idx: NightIndex, event: Event) -> None:
        if isinstance(event, Blockage):
            self._blockage_stack.append(event)
        else:
            try:
                self._events[night_idx][site].append(event)
            except KeyError:
                raise KeyError(f'Tried to add event {event} for inactive combination of site {site} and '
                               f'night index {night_idx}.')

    def add_events(self, site: Site, events: Iterable[Event], night_idx: NightIndex) -> None:
        for event in events:
            self.add_event(site, night_idx, event)

    def check_blockage(self, resume_event: ResumeNight) -> Blockage:
        if self._blockage_stack and len(self._blockage_stack) == 1:
            b = self._blockage_stack.pop()
            b.ends(resume_event.start)
            return b

        raise RuntimeError('Missing blockage for ResumeNight')

    def get_night_events(self, night_idx: int, site: Site) -> deque:

        return self._events.get(night_idx).get(site)
