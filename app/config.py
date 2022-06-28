from dataclasses import dataclass
from astropy.time import Time
from astropy.units import Quantity


@dataclass
class CollectorConfig:
    semesters: set
    program_types: set
    obs_classes: set


@dataclass
class SelectorConfig:
    properties: type


@dataclass
class SchedulerConfig:
    start_time: Time
    end_time: Time
    time_slot_length: Quantity
    sites: set
    collector: CollectorConfig
    selector: SelectorConfig
