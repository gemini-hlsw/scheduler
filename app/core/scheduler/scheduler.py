# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import logging
from typing import NoReturn

from astropy.time import Time

from app.config import config
from app import dispatcher
from app.core.scheduler.modes import SimulationMode, ValidationMode, OperationMode, SchedulerMode


class Scheduler:

    def __init__(self, start_time: Time, end_time: Time):
        self.start_time = start_time
        self.end_time = end_time
        self._mode = None
    
    def set_mode(self, mode: SchedulerMode) -> NoReturn:
        self._mode = mode

    def __call__(self):
        logging.info(f'Running on mode {self._mode}')
        self._mode.schedule(self.start_time, self.end_time)

@dispatcher.dispatch_with(config.mode)
def build_scheduler(start: Time = Time("2018-10-01 08:00:00", format='iso', scale='utc'),
                    end: Time = Time("2018-10-03 08:00:00", format='iso', scale='utc')) -> Scheduler:
    
    # Set scheduler mode based on config
    scheduler = Scheduler(start, end)
    return scheduler
