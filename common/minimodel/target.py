from abc import ABC
from dataclasses import dataclass
from enum import auto, Enum, IntEnum
from typing import Set

import numpy.typing as npt

from .magnitude import Magnitude

TargetName = str


class TargetType(Enum):
    """
    The type associated with a target in an observation.
    """
    BASE = auto()
    USER = auto()
    BLIND_OFFSET = auto()
    OFF_AXIS = auto()
    TUNING_STAR = auto()
    GUIDESTAR = auto()
    OTHER = auto()


class GuideSpeed(IntEnum):
    """
    How quickly a guider can guide on a guide star.
    """
    SLOW = auto()
    MEDIUM = auto()
    FAST = auto()


class TargetTag(Enum):
    """
    A tag used by nonsidereal targets to indicate their type.
    """
    COMET = auto()
    ASTEROID = auto()
    MAJOR_BODY = auto()


@dataclass
class Target(ABC):
    """
    Basic target information.
    """
    name: TargetName
    magnitudes: Set[Magnitude]
    type: TargetType

    def guide_speed(self) -> GuideSpeed:
        """
        Calculate the guide speed for this target.
        """
        ...


@dataclass
class SiderealTarget(Target):
    """
    For a SiderealTarget, we have an RA and Dec and then proper motion information
    to calculate the exact position.

    RA and Dec should be specified in decimal degrees.
    Proper motion must be specified in milliarcseconds / year.
    Epoch must be the decimal year.

    NOTE: The proper motion adjusted coordinates can be found in the TargetInfo in coord.
    """
    ra: float
    dec: float
    pm_ra: float
    pm_dec: float
    epoch: float


@dataclass
class NonsiderealTarget(Target):
    """
    For a NonsiderealTarget, we have a HORIZONS designation to indicate the lookup
    information, a tag to determine the type of target, and arrays of ephemerides
    to specify the position.
    """
    des: str
    tag: TargetTag
    ra: npt.NDArray[float]
    dec: npt.NDArray[float]
