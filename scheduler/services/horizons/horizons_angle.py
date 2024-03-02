# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC
from typing import final, Final

import numpy as np

from scheduler.services import logger_factory


__all__ = [
    'HorizonsAngle',
]


logger = logger_factory.create_logger(__name__)


@final
class HorizonsAngle(ABC):
    """
    This class should never be instantiated.
    It is simply a collection of static convenience methods for converting angles.
    """
    MICROARCSECS_PER_DEGREE: Final[float] = 60 * 60 * 1000 * 1000

    @staticmethod
    def to_signed_microarcseconds(angle: float) -> float:
        """
        Convert an angle in radians to a signed microarcsecond angle.
        """
        degrees = HorizonsAngle.to_degrees(angle)
        if degrees > 180:
            degrees -= 360
        return degrees * HorizonsAngle.MICROARCSECS_PER_DEGREE

    @staticmethod
    def to_degrees(angle: float) -> float:
        """
        Convert an angle in radians to a signed degree angle.
        """
        return angle * 180.0 / np.pi

    @staticmethod
    def to_microarcseconds(angle: float) -> float:
        """
        Convert an angle in radians to a signed microarcsecond angle.
        """
        return HorizonsAngle.to_degrees(angle) * HorizonsAngle.MICROARCSECS_PER_DEGREE
