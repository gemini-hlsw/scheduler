# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from enum import Enum
from typing import final

import strawberry  # noqa

from scheduler.core.sources.sources import Sources
from .scheduler_builder import SchedulerBuilder
from .validation_builder import ValidationBuilder
from ..events_queue import EventQueue


__all__ = [
    'SchedulerModes',
    'dispatch_with',
]


@final
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
