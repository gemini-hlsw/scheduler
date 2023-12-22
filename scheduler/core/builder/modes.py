# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from enum import Enum

import strawberry  # noqa

from scheduler.core.sources import Sources
from .schedulerbuilder import SchedulerBuilder
from .validationbuilder import ValidationBuilder
from ..eventsqueue import EventQueue


@strawberry.enum
class SchedulerModes(Enum):
    """Scheduler modes available:

    - OPERATION
    - SIMULATION
    - VALIDATION

    """
    OPERATION = 'operation'
    SIMULATION = 'simulation'
    VALIDATION = 'validation'


def dispatch_with(mode: SchedulerModes, sources: Sources, events: EventQueue) -> SchedulerBuilder:
    match mode:
        case SchedulerModes.VALIDATION:
            return ValidationBuilder(sources, events)
        case SchedulerModes.SIMULATION:
            raise ValueError(f'{mode.value} not implemented yet.')
        case SchedulerModes.OPERATION:
            raise ValueError(f'{mode.value} not implemented yet.')
