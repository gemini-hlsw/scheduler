from typing import FrozenSet

from common.minimodel import *


@dataclass
class Selector:
    """
    This is the Selector portion of the automated Scheduler.
    It selects the scheduling candidates that are viable for the data collected by
    the Collector.
    """

    start_time: datetime
    time_length: timedelta
    delta_time: timedelta = timedelta(minutes=1)
    sites: FrozenSet[Site] = frozenset(s for s in Site)

    def __post_init__(self):
        """
        Initialize internal non-input data members.
        """
        ...
