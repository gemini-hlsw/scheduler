# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime

import strawberry  # noqa
from strawberry.file_uploads import Upload  # noqa
from typing import Optional, List

from .scalars import Sites
from scheduler.core.builder.modes import SchedulerModes


@strawberry.input
class CreateNewScheduleInput:
    """
    Input for creating a new schedule.
    """
    start_time: str
    end_time: str
    num_nights_to_schedule: int
    sites: Sites
    mode: SchedulerModes


@strawberry.input
class UseFilesSourceInput:
    service: str
    site: Sites
    calendar: Optional[Upload] = None
    gmos_fpus: Optional[Upload] = None
    gmos_gratings: Optional[Upload] = None


@strawberry.input
class NewFault:
    reason: str
    instrument: str  # change to enum
    start: datetime  # for Fault event
    end: datetime  # for ResumeNight event


@strawberry.input
class AddEventInput:
    events: List[NewFault]
