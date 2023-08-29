# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import strawberry  # noqa
from strawberry.file_uploads import Upload
from typing import Optional
from .scalars import Sites
from scheduler.core.service.modes import SchedulerMode


@strawberry.input
class CreateNewScheduleInput:
    """
    Input for creating a new schedule.
    """
    start_time: str
    end_time: str
    num_nights_to_schedule: int
    site: Sites
    with_mode: SchedulerMode


@strawberry.input
class UseFilesSourceInput:
    service: str
    site: Sites
    calendar: Optional[Upload] = None
    gmos_fpus: Optional[Upload] = None
    gmos_gratings: Optional[Upload] = None
