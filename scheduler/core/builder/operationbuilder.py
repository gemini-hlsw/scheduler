# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import final, FrozenSet

from astropy.time import Time
from lucupy.minimodel import Site

from .schedulerbuilder import SchedulerBuilder


__all__ = [
    'OperationBuilder',
]


@final
class OperationBuilder(SchedulerBuilder):
    def _setup_event_queue(self,
                           start: Time,
                           num_nights_to_schedule: int,
                           sites: FrozenSet[Site]) -> None:
        """
        Set up the event queue for this particular mode of the Scheduler.
        """
        raise NotImplementedError('OperationsBuilder _setup_event_queue is not implemented.')
