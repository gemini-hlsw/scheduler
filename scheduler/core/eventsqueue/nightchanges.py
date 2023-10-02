from dataclasses import dataclass, field
from typing import Dict

from scheduler.core.eventsqueue import Event
from scheduler.core.plans import Plans


@dataclass
class NightChanges:
    lookup: Dict[Event, Plans] = field(default_factory=dict)

    def get_final_plans(self):
        return list(self.lookup.values())[-1]

