# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from typing import final, Optional

from lucupy.minimodel import NightIndex, TimeslotIndex


__all__ = [
    'TimeCoordinateRecord',
]


@final
@dataclass(frozen=True)
class TimeCoordinateRecord:
    """
    Time coordinate record calculated for an event.
    Attributes:
        night_idx: the night index for which the event was recorded
        timeslot_idx: the timeslot index for which the plan should be recalculated
        perform_time_accounting: True if time accounting should be performed, and False to suppress time accounting
        done: True if the night is done, and False otherwise
    """
    night_idx: Optional[NightIndex] = field(default=None)
    timeslot_idx: Optional[TimeslotIndex] = field(default=None)
    perform_time_accounting: bool = field(default=True)
    done: bool = field(default=False)
