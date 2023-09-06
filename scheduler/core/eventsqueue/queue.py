from collections import deque
from typing import List

from .events import Event, Blockage, ResumeNight


class EventQueue:
    def __init__(self, events: List[Event]):
        self.events = deque(events) if events else deque()
        self._blockage_stack = [e for e in events if isinstance(e, Blockage)]

    def _add(self, e: Event) -> None:
        if isinstance(e, Blockage):
            self._blockage_stack.append(e)
        else:
            self.events.append(e)

    def add_events(self, events: List[Event] | Event) -> None:
        if isinstance(events, list):
            for e in events:
                self._add(e)
        else:
            self._add(events)

    def check_blockage(self, resume_event: ResumeNight):
        if self._blockage_stack and len(self._blockage_stack) == 1:
            b = self._blockage_stack.pop()
            b.ends(resume_event.start)
            return b

        raise RuntimeError('Missing blockage for ResumeNight')
