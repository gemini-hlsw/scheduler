from collections import deque
from datetime import datetime
from typing import List

from .events import Event, Blockage, ResumeNight


class EventQueue:
    def __init__(self):
        self._events = {}
        self._blockage_stack = []

    def _add(self, e: Event, night_idx: int) -> None:
        if isinstance(e, Blockage):
            self._blockage_stack.append(e)
        else:
            if night_idx in self._events:
                self._events[night_idx].append(e)
            else:
                self._events[night_idx] = [e]

    def add_events(self, events: List[Event] | Event, night_idx: int) -> None:
        if isinstance(events, list):
            for e in events:
                self._add(e, night_idx)
        else:
            self._add(events, night_idx)

    def check_blockage(self, resume_event: ResumeNight) -> Blockage:
        if self._blockage_stack and len(self._blockage_stack) == 1:
            b = self._blockage_stack.pop()
            b.ends(resume_event.start)
            return b

        raise RuntimeError('Missing blockage for ResumeNight')

    def get_night_events(self, night_idx: int):
        return self._events[night_idx] if night_idx in self._events else None
