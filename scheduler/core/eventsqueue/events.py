from datetime import datetime, timedelta
from typing import List

from lucupy.minimodel import Resource, Conditions

from scheduler.core.meta import AbstractDataclass, FrozenAbstractDataclass


class Interruption(FrozenAbstractDataclass):
    """
    Parent class for any interruption in the night that would
    cause a new schedule to be created.
    """
    start: datetime
    reason: str


class Blockage(AbstractDataclass):
    """
    Parent class for any event in the night that would block
    time slots though the night.
    """
    start: datetime
    end: datetime = None # needs a resume event
    reason: str

    def ends(self, end: datetime) -> None:
        self.end = end

    def time_loss(self) -> timedelta:
        if self.end:
            return self.end - self.start
        else:
            raise ValueError("Can't calculate Blockage time loss without end value")


class ResumeNight(Interruption):
    """
    Event that let the scheduler knows that the night can be resumed
    by a new
    """
    pass


class Fault(Blockage):
    affects: List[Resource]


class WeatherChange(Interruption):
    new_conditions: Conditions


class Rto0s(Interruption):
    # value: ToO
    pass


Event = Interruption | Blockage
