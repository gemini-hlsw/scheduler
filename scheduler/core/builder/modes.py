# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause


from enum import Enum
from .builder import ValidationBuilder, SchedulerBuilder
from scheduler.core.sources import Sources


class SchedulerModes(Enum):
    """Scheduler modes available:

    - OPERATION
    - SIMULATION
    - VALIDATION

    """
    OPERATION = 'operation'
    SIMULATION = 'simulation'
    VALIDATION = 'validation'


def dispatch_with(mode: SchedulerModes, sources: Sources) -> SchedulerBuilder:
    match mode:
        case SchedulerModes.VALIDATION:
            return ValidationBuilder(sources)
        case SchedulerModes.SIMULATION:
            raise ValueError(f'{mode.value} not implemented yet.')
        case SchedulerModes.OPERATION:
            raise ValueError(f'{mode.value} not implemented yet.')

