# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from typing import final, Optional

from lucupy.minimodel import TimeslotIndex

from scheduler.core.eventsqueue.events import Event


__all__ = [
    'TimeCoordinateRecord',
]


@final
@dataclass(frozen=True)
class TimeCoordinateRecord:
    """
    Time coordinate record calculated for an event.
    Attributes:
        event: the event for which this time coordinate record holds
        timeslot_idx: the timeslot index for which the plan should be recalculated
        perform_time_accounting: True if time accounting should be performed, and False to suppress time accounting
        done: True if the night is done, and False otherwise
    """
    event: Event
    timeslot_idx: Optional[TimeslotIndex]
    perform_time_accounting: bool = field(default=True)
    done: bool = field(default=False)
