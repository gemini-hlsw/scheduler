# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import FrozenSet

from lucupy.minimodel import Resource, Conditions, Site


@dataclass
class Event(ABC):
    """
    Superclass for all events, i.e. Interruption and Blockage.
    """
    start: datetime
    reason: str
    site: Site


@dataclass
class Interruption(Event):
    """
    Parent class for any interruption that might cause a new schedule to be created.
    """
    ...


@dataclass
class Twilight(Interruption):
    """
    An event indicating that the 12 degree starting twilight for a night has been reached.
    """
    ...


@dataclass
class Blockage(Event):
    """
    Parent class for any interruption that causes a blockage and requires a resume event.
    """
    end: datetime = None  # needs a resume event

    def ends(self, end: datetime) -> None:
        self.end = end

    def time_loss(self) -> timedelta:
        if self.end:
            return self.end - self.start
        else:
            raise ValueError("Can't calculate Blockage time loss without end value.")


@dataclass
class ResumeNight(Interruption):
    """
    Event that lets the scheduler knows that the night can be resumed.
    """
    ...


class Fault(Blockage):
    """
    Blockage that occurs when one or more resources experience a fault.
    """
    affects: FrozenSet[Resource]


@dataclass
class WeatherChange(Interruption):
    """
    Interruption that occurs when new weather conditions come in.
    """
    new_conditions: Conditions
