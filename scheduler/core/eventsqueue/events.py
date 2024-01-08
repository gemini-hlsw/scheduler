# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import uuid
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import final, FrozenSet

from lucupy.minimodel import Resource, Conditions, TimeslotIndex
from lucupy.timeutils import time2slots


@dataclass
class UUIDIdentified(ABC):
    """
    A class with an automatic UUID attached to it.
    TODO: This should be moved to lucupy.
    """
    id: uuid.UUID = field(default_factory=uuid.uuid4, init=False)

    def __eq__(self, other):
        if isinstance(other, UUIDIdentified):
            return self.id == other.id
        return False


@dataclass
class UUIDReferenced(ABC):
    """
    A class for an object that maintains a reference to a UUIDIdentified object.
    """
    uuid_identified: UUIDIdentified

    @property
    def uuid_referenced(self) -> uuid:
        return self.uuid_identified.id


@dataclass
class Event(UUIDIdentified, ABC):
    """
    Superclass for all events. They contain:
    1. The time (as a datetime object) at which an event occurred.
    2. A human-readable description of the event.
    """
    time: datetime
    description: str

    def to_timeslot_idx(self, twi_eve_time: datetime, time_slot_length: timedelta) -> TimeslotIndex:
        """
        Given an event, calculate the timeslot offset it falls into relative to another datetime.
        This would typically be the twilight of the night on which the event occurs, hence the name twi_eve_time.
        """
        time_from_twilight = self.time - twi_eve_time
        time_slots_from_twilight = time2slots(time_slot_length, time_from_twilight)
        return TimeslotIndex(time_slots_from_twilight)


@dataclass
class RoutineEvent(Event, ABC):
    """
    A routine event that is predictable and processed by the Scheduler.
    Examples include evening and morning twilight.
    """
    ...


@dataclass
class TwilightEvent(RoutineEvent, ABC):
    """
    An event indicating that the 12 degree starting twilight for a night has been reached.
    """
    ...


@final
@dataclass
class EveningTwilightEvent(TwilightEvent):
    """
    An event indicating that the 12 degree starting twilight for a night has been reached.
    """
    ...


@final
@dataclass
class MorningTwilightEvent(TwilightEvent):
    """
    An event indicating that the 12 degree morning twilight for a night has been reached.
    This is used to finalize the time accounting for the night.
    """
    ...


@dataclass
class InterruptionEvent(Event, ABC):
    """
    Parent class for any interruption that might cause a new schedule to be created.
    These events include:
    1. Events that have no specified end time (e.g. weather changes).
    2. Events that have a specified end time (e.g. engineering tasks, faults) and thus are paired together
       with an InterruptionResolutionEvent.
    """
    ...


@final
@dataclass
class WeatherChangeEvent(InterruptionEvent):
    """
    Interruption that occurs when new weather conditions come in.
    """
    new_conditions: Conditions


@final
@dataclass
class FaultEvent(InterruptionEvent):
    """
    Interruption that occurs when there is a fault in a resource.
    TODO: Should we have one per Resource (probably), or one for multiple resources?
    """
    affects: FrozenSet[Resource]


@final
@dataclass
class EngineeringTaskEvent(InterruptionEvent):
    """
    A class representing software engineering tasks.
    """
    ...


@dataclass
class InterruptionResolutionEvent(InterruptionEvent, UUIDReferenced, ABC):
    """
    A class representing the resolution of an interruption that can be resolved (e.g. a resolved fault
    or the end of an engineering task.)

    These events, signifying the end of a period of time, can be used to generate a time loss.
    """
    @property
    def time_loss(self) -> timedelta:
        """
        Calculate the time loss from this InterruptionEvent to this InterruptionEventResolution as a timedelta.
        """
        if not isinstance(self.uuid_identified, InterruptionEvent):
            raise ValueError(f'{self.__class__.__name__} ({self.description}) does not have a corresponding '
                             'interruption event.')
        return self.time - self.uuid_identified.time

    def time_slot_loss(self, time_slot_length: timedelta) -> int:
        """
        Given the length of a time slot, calculate the number of time slots lost from the InterruptionEvent to
        this InterruptionEventResolution.
        """
        return time2slots(time_slot_length, self.time_loss)


@final
class FaultResolutionEvent(InterruptionEvent, UUIDReferenced):
    """
    Interruption that occurs when a Fault is resolved.
    """
    def __post_init__(self):
        if not isinstance(self.uuid_identified, FaultEvent):
            raise ValueError('FaultResolutionEvent is not paired with a FaultEvent.')


@dataclass
class EngineeringTaskResolutionEvent(InterruptionEvent, UUIDReferenced):
    """
    Interruption that occurs when an EngineeringTask is completed.
    """
    def __post_init__(self):
        if not isinstance(self.uuid_identified, EngineeringTaskEvent):
            raise ValueError('EngineeringTaskResolutionEvent is not paired with an EngineeringTaskEvent.')