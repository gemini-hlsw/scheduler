# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from typing import final

import numpy as np


__all__ = [
    'Coordinates',
]


@final
@dataclass(frozen=True)
class Coordinates:
    """
    Both ra and dec are in radians.
    """
    ra: float
    dec: float

    def angular_distance(self, other: 'Coordinates') -> float:
        """
        Calculate the angular distance between two points on the sky in radians.
        Code is based on
        https://github.com/gemini-hlsw/lucuma-core/blob/master/modules/core/shared/src/main/scala/lucuma/core/math/Coordinates.scala#L52
        """
        delta_ra = other.ra - self.ra
        delta_dec = other.dec - self.dec
        a = np.sin(delta_dec / 2) ** 2 + np.cos(self.dec) * np.cos(other.dec) * np.sin(delta_ra / 2) ** 2
        return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def interpolate(self, other: 'Coordinates', f: float) -> 'Coordinates':
        """
        Interpolate between two Coordinates objects.
        """
        delta = self.angular_distance(other)
        if delta == 0:
            return Coordinates(self.ra, self.dec)
        else:
            a = np.sin((1 - f) * delta) / np.sin(delta)
            b = np.sin(f * delta) / np.sin(delta)
            x = a * np.cos(self.dec) * np.cos(self.ra) + b * np.cos(other.dec) * np.cos(other.ra)
            y = a * np.cos(self.dec) * np.sin(self.ra) + b * np.cos(other.dec) * np.sin(other.ra)
            z = a * np.sin(self.dec) + b * np.sin(other.dec)
            phi_i = np.arctan2(z, np.sqrt(x * x + y * y))
            lambda_i = np.arctan2(y, x)
            return Coordinates(lambda_i, phi_i)
