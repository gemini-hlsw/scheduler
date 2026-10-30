# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from datetime import datetime, timedelta
from typing import List, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from lucupy import sky
from lucupy.minimodel import Site

__all__ = ['moon_altitudes', 'moon_phase']


def moon_altitudes(
    site: Site, start: datetime, end: datetime, step: timedelta = timedelta(minutes=5)
) -> List[Tuple[datetime, float]]:
    """
    The moon's altitude at a site, every step from start to end, both included.

    Computed the way NightEvents computes it for the scheduler, but over any range, e.g. past the twilights.

    Returns:
        (time in UTC, altitude in degrees) pairs.
    """
    n = int((end - start) / step) + 1
    times = Time(start) + np.arange(n) * step.total_seconds() * u.s
    pos, _ = sky.Moon().at(times).accurate_location(site.location)
    lst = sky.local_sidereal_time(times, site.location)
    alt, _, _ = sky.Altitude.above(pos.dec, lst - pos.ra, site.location.lat)
    return [(start + i * step, float(a)) for i, a in enumerate(alt.deg)]


def moon_phase(site: Site, time: datetime) -> Tuple[float, bool]:
    """
    How much of the moon is lit, seen from a site, and whether it is waxing.

    Returns:
        The illuminated fraction, from 0 (new) to 1 (full), and True while waxing.
    """
    t = Time(time)
    sun = SkyCoord(sky.Sun.at(t))
    moon, _ = sky.Moon().at(t).accurate_location(site.location)
    illumination = (1 - np.cos(sun.separation(moon).rad)) / 2
    # Waxing while the moon is east of the sun, i.e. up to 180° ahead in right ascension.
    waxing = (moon.ra - sun.ra).wrap_at(360 * u.deg) < 180 * u.deg
    return float(illumination), bool(waxing)
