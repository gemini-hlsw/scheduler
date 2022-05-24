from .atom import *
from .constraints import *
from .group import *
from .magnitude import *
from .observation import *
from .program import *
from .qastate import *
from .resource import *
from .semester import *
from .site import *
from .target import *
from .timeallocation import *
from .timingwindow import *
from .too import *

# Type alias for night indices.
NightIndex = int

class Plans:
    """
    A collection of Nightly Plan from a corresponding Site
    """
    def __init__(self, night_events):
        self.durations = [len(night_events.times[idx]) for idx, _ in enumerate(night_events.time_grid)]
        self.start_times = [night_events.local_times[idx][0] for idx, _ in enumerate(night_events.time_grid)]
        self.end_times = [night_events.local_times[idx][-1] for idx, _ in enumerate(night_events.time_grid)]
        self.time_slot_length = night_events.time_slot_length

    def __iter__(self):
        self.plans = []
        return self

    def __next__(self):
        if len(self.plans) <= len(self.durations):
            self.plans.append(Plan(next(iter(self.start_times)),
                                   next(iter(self.end_times)),
                                   slot_length=self.time_slot_length,
                                   night_duration=next(iter(self.durations))))
            return next(iter(self.plans))
        raise StopIteration


class Plan:
    """
    A 'plan' is a collection of nighly plans
    """
    def __init__(self, start, end, slot_length=1, night_duration=10):
        self.start = start
        self.end = end
        self._time_slot_length = slot_length
        self._time_slots_left = night_duration
        self._visits = []
    
    def _time2slots(self, time) -> int:
        return ceil((time.total_seconds() / 60) / self._time_slot_length.value)

    def add_group(self, group):
        self._visits.append(group)
        self._time_slots_left -= self._time2slots(group[0].total_used())
    
    def is_full(self):
        return self._time_slots_left <= 0

    def time_slots_left(self):
        return self._time_slots_left
