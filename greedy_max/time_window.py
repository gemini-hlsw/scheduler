from dataclasses import dataclass
from typing import Dict
from numpy import ndarray 
from greedy_max.site import Site

@dataclass
class TimeWindow:
    """
    A single time window that defines the boundary where an observation can be scheduled.
    Contains also the calibration time, total observation time and the time intervals in the schedule
    for that observation.
    """
    start: int
    end: int
    length: int
    time_slots: int
    indices: ndarray
    min_slot_time: int
    intervals: Dict[Site, ndarray]