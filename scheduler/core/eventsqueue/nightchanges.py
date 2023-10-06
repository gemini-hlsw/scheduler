from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Tuple, List

import numpy as np
from lucupy.types import Interval

from scheduler.core.eventsqueue import Event
from scheduler.core.plans import Plans


@dataclass
class NightChanges:
    lookup: Dict[Event, Plans] = field(default_factory=dict)

    def get_final_plans(self):
        return list(self.lookup.values())[-1]


class TimelineCode(Enum):
    AVAILABLE = auto()
    SCHEDULE = auto()
    WEATHERCHANGE = auto()
    FAULT = auto()
