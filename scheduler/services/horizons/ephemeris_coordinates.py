# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, final

import numpy as np
import numpy.typing as npt

from scheduler.services import logger_factory
from .coordinates import Coordinates


__all__ = [
    'EphemerisCoordinates',
]


logger = logger_factory.create_logger(__file__)


@final
@dataclass(frozen=True)
class EphemerisCoordinates:
    """
    Both ra and dec are in radians.

    """
    coordinates: List[Coordinates]
    time: npt.NDArray[float]

    def _bracket(self, time: datetime) -> Tuple[datetime, datetime]:
        """
        Return both lower and upper of the given time: i.e., the closest elements on either side.
        """
        return self.time[self.time > time].min(), self.time[self.time < time].max()

    def interpolate(self, time: datetime) -> Coordinates:
        """
        Interpolate ephemeris to a given time.
        """
        a, b = self._bracket(time)
        # Find indexes for each bound
        i_a, i_b = np.where(self.time == a)[0][0], np.where(self.time == b)[0][0]
        factor = (time.timestamp() - a.timestamp() / b.timestamp() - a.timestamp()) * 1000
        logger.info(f'Interpolating by factor: {factor}')

        return self.coordinates[i_a].interpolate(self.coordinates[i_b], factor)
