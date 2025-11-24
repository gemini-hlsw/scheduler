# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from enum import Enum
from typing import final

import strawberry  # noqa

from scheduler.core.sources.sources import Sources
from . import SimulationBuilder
from .schedulerbuilder import SchedulerBuilder
from .validationbuilder import ValidationBuilder
from scheduler.core.events.queue import EventQueue
from scheduler.config import config


__all__ = [
    'SchedulerModes',
    'dispatch_with',
]

from ..sources import Origins


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

try:
    app_mode = SchedulerModes(config.app.mode)
except ValueError:
    app_mode = SchedulerModes.VALIDATION

def dispatch_with(mode: SchedulerModes, sources: Sources, events: EventQueue) -> SchedulerBuilder:
    match mode:
        case SchedulerModes.VALIDATION:
            sources.set_origin(Origins.OCS())
            return ValidationBuilder(sources, events)
        case SchedulerModes.SIMULATION:
            sources.set_origin(Origins.SIM())
            return SimulationBuilder(sources, events)
        case SchedulerModes.OPERATION:
            raise ValueError(f'{mode.value} not implemented yet.')
