# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import FrozenSet

from scheduler.services.resource.filters import AbstractFilter

from lucupy.minimodel import Resource, Site


@dataclass(frozen=True)
class Interruption:
    """
    Parent class for any interruption in the night that would
    cause missing time of observation.
    """
    start: datetime
    time_loss: timedelta
    reason: str


@dataclass(frozen=True)
class Fault(Interruption):
    id: str


@dataclass(frozen=True)
class EngTask(Interruption):
    end: datetime


# An instance of this class exists for every night in the configuration file.
@dataclass(frozen=True)
class NightConfiguration:
    site: Site
    local_date: date

    # Is the night a laser night?
    is_lgs: bool

    # Are we accepting ToOs?
    too_status: bool

    # This is a filter to determine:
    # 1. Highest priority programs.
    # 2. Prohibited programs.
    # 3. Highest priority scheduling groups.
    # 4. Prohibited scheduling groups.
    filter: AbstractFilter

    # The list of resources available for the night.
    resources: FrozenSet[Resource]

    # List of faults that happened in the night causing time losses.
    # faults: FrozenSet[Fault]

    # List of Engineering Task, this would block part of the night.
    # Some are bound in the twilight so should be completed in Collector.
    # eng_tasks: FrozenSet[EngTask]
