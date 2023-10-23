# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from collections import deque
from typing import FrozenSet, Iterable

from lucupy.minimodel import Site

from .events import Event, Blockage, ResumeNight


class EventQueue:
    def __init__(self, night_indices: FrozenSet[int], sites: FrozenSet[Site]):
        self._events = {n_idx: {site: deque([]) for site in sites} for n_idx in night_indices}
        self._blockage_stack = []

    def add_event(self, e: Event, night_idx: int, site: Site) -> None:
        if isinstance(e, Blockage):
            self._blockage_stack.append(e)
        else:
            try:
                self._events[night_idx][site].append(e)
            except KeyError:
                raise KeyError(f"NightIndex {night_idx} or Site {site} doesn't exist")

    def add_events(self, site: Site, events: Iterable[Event], night_idx: int) -> None:
        for event in events:
            self.add_event(event, night_idx, site)

    def check_blockage(self, resume_event: ResumeNight) -> Blockage:
        if self._blockage_stack and len(self._blockage_stack) == 1:
            b = self._blockage_stack.pop()
            b.ends(resume_event.start)
            return b

        raise RuntimeError('Missing blockage for ResumeNight')

    def get_night_events(self, night_idx: int, site: Site) -> deque:

        return self._events.get(night_idx).get(site)
