# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from datetime import datetime
from typing import final, List

import numpy as np
import numpy.typing as npt

from .coordinates import Coordinates

__all__ = [
    "EphemerisCoordinates"
]


@final
@dataclass(frozen=True)
class EphemerisCoordinates:
    coordinates: List[Coordinates]
    time: npt.NDArray[float]

    def interpolate(self, time: datetime) -> Coordinates:
        lower = self.time[self.time > time].min()
        lower_ts = lower.timestamp()
        lower_idx = np.where(self.time == lower)[0][0]

        upper = self.time[self.time < time].max()
        upper_ts = upper.timestamp()
        upper_idx = np.where(self.time == upper)[0][0]
        print(f'bracket: input={time}, lower={lower}, upper={upper}, lower_idx={lower_idx}, upper_idx={upper_idx}')

        # Find indexes for each bound.
        factor = (time.timestamp() - lower_ts) / (upper_ts - lower_ts) * 1000
        print(f'Interpolating by factor: {factor}')

        return self.coordinates[lower_idx].interpolate(self.coordinates[upper_idx], factor)
