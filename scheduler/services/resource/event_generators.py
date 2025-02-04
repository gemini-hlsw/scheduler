# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import abstractmethod, ABC
from dataclasses import dataclass, field, InitVar
from datetime import datetime
from typing import FrozenSet, Optional

from lucupy.minimodel import Resource, Site, ObservationID

from scheduler.core.eventsqueue.events import (FaultEvent, FaultResolutionEvent,
                                               InterruptionEvent, InterruptionResolutionEvent,
                                               WeatherClosureEvent, WeatherClosureResolutionEvent, ToOActivationEvent)


__all__ = [
    'EngineeringTask',
    'Interruption',
    'Fault',
    'WeatherClosure',
]


@dataclass(frozen=True)
class EngineeringTask:
    """
    The information record for an Engineering Task.
    Note that Engineering tasks are planned and are not "surprises" to the Scheduler.
    """
    site: Site
    start_time: datetime
    end_time: datetime
    description: str


@dataclass(frozen=True)
class Interruption(ABC):
    """
    The superclass information record for an interruption, which is a "surprise" to the Scheduler.
    Note that this class is only used by OCS services and simulations. It will not work for GPP.

    It relies on knowing the duration of the interruption, which is known in historical OCS data and in simulated
    data. For GPP, another technique will have to be used as resolutions will stream in when done.
    TODO: Determine with GPP team how resolutions to faults and weather closures will be dispatched.
    TODO: Asked in #GPP slack on 2024-02-29,
    """
    site: Site
    start_time: datetime
    end_time: datetime
    description: str
    affected_resources: InitVar[Optional[FrozenSet[Resource]]] = None
    _affected_resources: FrozenSet[Resource] = field(init=False, default_factory=frozenset)

    @property
    def affects(self) -> FrozenSet[Resource]:
        """
        An interruption to one or more Resources that prevents them from being used until the interruption is resolved.
        This depends upon the type of Interruption.
        """
        return self._affected_resources

    def __post_init__(self, affected_resources: FrozenSet[Resource]):
        if not affected_resources:
            affected_resources = frozenset([self.site.resource])
        object.__setattr__(self, '_affected_resources', affected_resources)

    @abstractmethod
    def to_events(self) -> (InterruptionEvent, InterruptionResolutionEvent):
        """
        Convert the data herein to a pair of events,
        """
        ...


@dataclass(frozen=True)
class Fault(Interruption):
    ...

    def to_events(self) -> (FaultEvent, FaultResolutionEvent):
        if self.description is not None:
            fault_description = f'Fault: {self.description}'
            resolution_description = f'Fault resolved: {self.description}'
        else:
            fault_description = 'Fault'
            resolution_description = 'Fault resolved'
        fault_event = FaultEvent(time=self.start_time,
                                 description=fault_description,
                                 affects=self.affects,
                                 site=self.site)
        fault_resolution_event = FaultResolutionEvent(time=self.end_time,
                                                      description=resolution_description,
                                                      uuid_identified=fault_event,
                                                      site=self.site)
        return fault_event, fault_resolution_event


@dataclass(frozen=True)
class WeatherClosure(Interruption):
    ...

    def to_events(self) -> (WeatherClosureEvent, WeatherClosureResolutionEvent):
        if self.description is not None:
            closure_description = f'Closure: {self.description}'
            resolution_description = f'Closure resolved: {self.description}'
        else:
            closure_description = 'Closure'
            resolution_description = 'Closure resolved'
        closure_event = WeatherClosureEvent(time=self.start_time,
                                            description=closure_description,
                                            site=self.site)
        closure_resolution_event = WeatherClosureResolutionEvent(time=self.end_time,
                                                                 description=resolution_description,
                                                                 uuid_identified=closure_event,
                                                                 site=self.site)
        return closure_event, closure_resolution_event


@dataclass(frozen=True)
class ToOActivation:
    site: Site
    too_id: ObservationID
    start_time: datetime

    def to_event(self) -> ToOActivationEvent:
        return ToOActivationEvent(site=self.site,
                                  time=self.start_time,
                                  too_id=self.too_id,
                                  description=f'Activation for {self.too_id.id}')
