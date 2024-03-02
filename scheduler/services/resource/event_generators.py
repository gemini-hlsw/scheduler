# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import FrozenSet

from lucupy.minimodel import Resource

from scheduler.core.events_queue.events import (EngineeringTaskEvent, EngineeringTaskResolutionEvent,
                                                FaultEvent, FaultResolutionEvent)

__all__ = ['EngineeringTask', 'Fault']


@dataclass(frozen=True)
class EngineeringTask:
    """
    The information record for an Engineering Task.
    """
    start_time: datetime
    end_time: datetime
    description: str

    def to_events(self) -> (EngineeringTaskEvent, EngineeringTaskResolutionEvent):
        eng_task_event = EngineeringTaskEvent(time=self.start_time,
                                              description=self.description)
        eng_task_resolution_event = EngineeringTaskResolutionEvent(time=self.end_time,
                                                                   description=f'Completed: {self.description}',
                                                                   uuid_identified=eng_task_event)
        return eng_task_event, eng_task_resolution_event


@dataclass(frozen=True)
class Fault:
    """
    The information record for a historic Fault, i.e. one where we already know the duration and the effects.
    """
    time: datetime
    duration: timedelta
    description: str

    # Assume unless otherwise stated that a fault affects no resources
    affects: FrozenSet[Resource] = field(default_factory=lambda: frozenset())

    def to_events(self) -> (FaultEvent, FaultResolutionEvent):
        fault_event = FaultEvent(time=self.time,
                                 description=self.description,
                                 affects=self.affects)
        fault_resolution_event = FaultResolutionEvent(time=self.time + self.duration,
                                                      description=f'Resolved: {self.description}',
                                                      uuid_identified=fault_event)
        return fault_event, fault_resolution_event
