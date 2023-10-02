from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from lucupy.minimodel import Resource, Conditions

from scheduler.core.meta import AbstractDataclass, FrozenAbstractDataclass


@dataclass(frozen=True)
class Interruption(FrozenAbstractDataclass):
    """
    Parent class for any interruption in the night that would
    cause a new schedule to be created.
    """
    start: datetime
    reason: str

    def __str__(self):
        return self.__class__.__name__


@dataclass
class Blockage(AbstractDataclass):
    """
    Parent class for any event in the night that would block
    time slots though the night.
    """
    start: datetime
    reason: str
    end: datetime = None  # needs a resume event

    def ends(self, end: datetime) -> None:
        self.end = end

    def time_loss(self) -> timedelta:
        if self.end:
            return self.end - self.start
        else:
            raise ValueError("Can't calculate Blockage time loss without end value")

    def __str__(self):
        return self.__class__.__name__


class ResumeNight(Interruption):
    """
    Event that let the scheduler knows that the night can be resumed
    by a new
    """
    pass


class Fault(Blockage):
    affects: List[Resource]


@dataclass(frozen=True)
class WeatherChange(Interruption):
    new_conditions: Conditions


class Rto0s(Interruption):
    # value: ToO
    pass


Event = Interruption | Blockage
