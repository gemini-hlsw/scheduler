from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar, Optional


@dataclass(frozen=True)
class TimingWindow:
    """
    Representation of timing windows in the mini-model.

    For infinite duration, set duration to timedelta.max.
    For repeat, -1 means forever repeating, 0 means non-repeating.
    For period, None should be used if repeat < 1.
    """
    start: datetime
    duration: timedelta
    repeat: int
    period: Optional[timedelta]

    # For infinite duration, use the length of an LP.
    INFINITE_DURATION_FLAG: ClassVar[int] = -1
    INFINITE_DURATION: ClassVar[int] = timedelta(days=3 * 365, hours=24)
    FOREVER_REPEATING: ClassVar[int] = -1
    NON_REPEATING: ClassVar[int] = 0
    NO_PERIOD: ClassVar[Optional[timedelta]] = None

    # A number to be used by the Scheduler to represent infinite repeats from the
    # perspective of the OCS: if FOREVER_REPEATING is selected, then it is converted
    # into this for calculation purposes.
    OCS_INFINITE_REPEATS: ClassVar[int] = 1000
