# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass

from numpy import arctan2, cos, sin, sqrt

__all__ = [
    'Coordinates'
]


@dataclass(frozen=True)
class Coordinates:
    """
    Both ra and dec must be in radians.
    """
    ra: float
    dec: float

    def angular_distance(self, other: 'Coordinates') -> float:
        delta_ra = other.ra - self.ra
        delta_dec = other.dec - self.dec
        a = sin(delta_dec / 2) ** 2 + cos(self.dec) * cos(other.dec) * sin(delta_ra / 2) ** 2
        dist = 2 * arctan2(sqrt(a), sqrt(1 - a))
        return dist

    def interpolate(self, other: 'Coordinates', ratio: float) -> 'Coordinates':
        """
        Interpolate between self and other for a ratio in [0.0, 1.0].
        """
        delta = self.angular_distance(other)
        if delta == 0:
            return self
        a = sin((1 - ratio) * delta) / sin(delta)
        b = sin(ratio * delta) / sin(delta)
        x = a * cos(self.dec) * cos(self.ra) + b * cos(other.dec) * cos(other.ra)
        y = a * cos(self.dec) * sin(self.ra) + b * cos(other.dec) * sin(other.ra)
        z = a * sin(self.dec) + b * sin(other.dec)
        phi_i = arctan2(z, sqrt(x * x + y * y))
        lambda_i = arctan2(y, x)
        return Coordinates(lambda_i, phi_i)
