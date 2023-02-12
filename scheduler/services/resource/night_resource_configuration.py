# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from datetime import date
from enum import Enum, auto
from typing import FrozenSet

from filters import AbstractFilter

from lucupy.minimodel import Resource, Site


class TelescopeMode(Enum):
    QUEUE = auto()
    PV = auto()
    CLASSICAL = auto()
    PRIORITY = auto()
    VISITOR_INSTRUMENT = auto()
    PARTNER_BLOCK = auto()


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
