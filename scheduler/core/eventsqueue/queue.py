from collections import deque
from datetime import datetime
from typing import List, FrozenSet
from collections import deque

from lucupy.minimodel import Site

from .events import Event, Blockage, ResumeNight


class EventQueue:
    def __init__(self, night_indices: List[int], sites: FrozenSet[Site]):
        self._events = {n_idx: {site: deque([]) for site in sites} for n_idx in night_indices}
        self._blockage_stack = []

    def _add(self, e: Event, night_idx: int, site: Site) -> None:
        if isinstance(e, Blockage):
            self._blockage_stack.append(e)
        else:
            try:
                self._events[night_idx][site].append(e)
            except KeyError:
                raise KeyError(f"NightIndex {night_idx} or Site {site} doesn't exist")

    def add_events(self, site: Site, events: List[Event] | Event, night_idx: int) -> None:
        if isinstance(events, list):
            for e in events:
                self._add(e, night_idx, site)
        else:
            self._add(events, night_idx, site)

    def check_blockage(self, resume_event: ResumeNight) -> Blockage:
        if self._blockage_stack and len(self._blockage_stack) == 1:
            b = self._blockage_stack.pop()
            b.ends(resume_event.start)
            return b

        raise RuntimeError('Missing blockage for ResumeNight')

    def get_night_events(self, night_idx: int, site: Site) -> deque:

        return self._events.get(night_idx).get(site)
