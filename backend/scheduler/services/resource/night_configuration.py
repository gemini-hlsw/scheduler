# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from datetime import date
from typing import FrozenSet

from lucupy.minimodel import Resource, Site

from .event_generators import EngineeringTask
from .filters import AbstractFilter

__all__ = ['NightConfiguration']


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

    # Historical faults that happened during the night.
    # faults: FrozenSet[Fault]

    # Historical engineering tasks that block part of the night.
    eng_tasks: FrozenSet[EngineeringTask]

    def __post_init__(self):
        """
        Calculate values once to avoid recalculation.
        """
        # We have to iterate over the eng_tasks and determine what time slots they cover.
        # For this, we need the night events to get the length of the night.
        ...
