from typing import FrozenSet

from collector import Collector
from common.minimodel import *
from common.scheduler import SchedulerComponent


@dataclass
class Selector(SchedulerComponent):
    """
    This is the Selector portion of the automated Scheduler.
    It selects the scheduling candidates that are viable for the data collected by
    the Collector.
    """
    collector: Collector

    def __post_init__(self):
        """
        Initialize internal non-input data members.
        """
        ...
