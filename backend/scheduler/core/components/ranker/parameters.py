# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, final

import astropy.units as u
from astropy.coordinates import Angle
import numpy as np
import numpy.typing as npt
from lucupy.minimodel import ALL_SITES, Band, Site
from lucupy.types import MinMax

__all__ = [
    'RankerParameters',
    'RankerBandParameters',
    'RankerBandParameterMap',
    'default_band_params',
]


def _default_score_combiner(x: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    The default function used to combine scores for Groups.
    """
    # Note we need to use 0. or applying this function results in an array of int instead of float.
    return np.array([np.max(x)]) if 0 not in x else np.array([0.])


# Default telescope altitude limits
# _def_alt_limits_site: Dict[Site, Dict[MinMax, Angle]] = {
#     Site.GS: {MinMax.MIN: Angle(18.0 * u.deg), MinMax.MAX: Angle(88.0 * u.deg)},
#     Site.GN: {MinMax.MIN: Angle(18.0 * u.deg), MinMax.MAX: Angle(88.0 * u.deg)}
# }

_def_alt_limits: Dict[MinMax, Angle] = {MinMax.MIN: Angle(18.0 * u.deg), MinMax.MAX: Angle(88.0 * u.deg)}

# These are not used in the current implementation
# _default_user_priority_factors: Dict[Priority, float] = {Priority.LOW: 1.0, Priority.MEDIUM: 1.25, Priority.HIGH: 1.5}


@final
@dataclass
class RankerParameters:
    """
    Global parameters for the Ranker.
    """
    thesis_factor: float = 1.1
    power: int = 2
    met_power: float = 1.0   # DefaultRanker: exponent. AdditiveRanker: linear weight.
    vis_power: float = 1.0   # DefaultRanker: exponent. AdditiveRanker: linear weight.
    wha_power: float = 1.0   # DefaultRanker: exponent. AdditiveRanker: linear weight.
    air_power: float = 0.0   # Always an exponent: score /= min(airmass)**air_power.
    program_priority: float = 10.0
    priority_factor: float = 8.0
    preimaging_factor: float = 1.25
    ongoing_factor: float = 1.5
    # altitude_limits: Dict[Site, Dict[MinMax, Angle]] = field(default_factory=lambda: _def_alt_limits_site)
    gs_altitude_limits: Dict[MinMax, Angle] = field(default_factory=lambda: _def_alt_limits)
    gn_altitude_limits: Dict[MinMax, Angle] = field(default_factory=lambda: _def_alt_limits)
    altitude_limits: Dict[Site, Dict[MinMax, Angle]] = field(init={})

    # user_priority_factors: Dict[Priority, float] = field(default_factory=lambda: _default_user_priority_factors)

    # HA weighting for zenith distances < 40 deg
    # Weighted to slightly positive HA, this was the original intention
    # dec_diff_less_40: npt.NDArray[float] = field(default_factory=lambda: np.array([3., 0.1, -0.06]))
    # Weighted to the meridian, used most of the time but backwards from the initial intention
    dec_diff_less_40: npt.NDArray[float] = field(default_factory=lambda: np.array([3., 0., -0.08]))

    # HA weighting for zenith distances > 40 deg (Xmin > 1.3)
    # Weighted to 0, the original intention
    dec_diff: npt.NDArray[float] = field(default_factory=lambda: np.array([3., 0., -0.08]))
    # Weighted to slightly positive HA, backwards from the intention
    # dec_diff: npt.NDArray[float] = field(default_factory=lambda: np.array([3., 0.1, -0.06]))

    score_combiner: Callable[[npt.NDArray[float]], npt.NDArray[float]] = field(init=False)

    def __post_init__(self):
        self.score_combiner = _default_score_combiner

        self.altitude_limits = {Site.GS: self.gs_altitude_limits, Site.GN: self.gn_altitude_limits}

        for site in ALL_SITES:
            if self.altitude_limits[site][MinMax.MIN] < Angle(18.0*u.deg):
                raise ValueError(f'The minimum altitude limit for {site.name} must be at least 18 degrees.')
            if self.altitude_limits[site][MinMax.MAX] > Angle(90.0*u.deg):
                raise ValueError(f'The maximum altitude limit for {site.name} must be 90 degrees or less.')

    def __altitude_limits_to_str(self) -> str:
        text = ""
        for idx, site in enumerate(self.altitude_limits):
            if idx != len(self.altitude_limits) - 1:
                text += "\n    ├─" + site.site_name + ": "
                for midx, minmax in enumerate(self.altitude_limits[site]):
                    if midx != len(self.altitude_limits[site]) - 1:
                        text += "\n    │ ├─" + str(minmax.name) + ": " + str(self.altitude_limits[site][minmax].value) + " deg"
                    else:
                        text += "\n    │ └─" + str(minmax.name) + ": " + str(self.altitude_limits[site][minmax].value) + " deg"
            else:
                text += "\n    └─" + site.site_name + ": "
                for midx, minmax in enumerate(self.altitude_limits[site]):
                    if midx != len(self.altitude_limits[site]) - 1:
                        text += "\n      ├─" + str(minmax.name) + ": " + str(self.altitude_limits[site][minmax].value) + " deg"
                    else:
                        text += "\n      └─" + str(minmax.name) + ": " + str(self.altitude_limits[site][minmax].value) + " deg"
        return text

    def __str__(self) -> str:
        return "Ranker Parameters\n" + \
        f"  ├─thesis_factor: {self.thesis_factor}\n" + \
        f"  ├─power: {self.power}\n" + \
        f"  ├─met_power: {self.met_power}\n" + \
        f"  ├─vis_power: {self.vis_power}\n" + \
        f"  ├─wha_power: {self.wha_power}\n" + \
        f"  ├─program_priority: {self.program_priority}\n" + \
        f"  ├─priority_factor: {self.priority_factor}\n" + \
        f"  ├─preimaging_factor: {self.preimaging_factor}\n" + \
        f"  └─altitude_limits: {self.__altitude_limits_to_str()}"

@final
@dataclass(frozen=True)
class RankerBandParameters:
    """
    Parameters per band for the Ranker.
    """
    m1: float
    b1: float
    m2: float
    b2: float
    xb: float
    xb0: float
    xc0: float


# A map of parameters per band for the Ranker.
RankerBandParameterMap = Mapping[Band, RankerBandParameters]


def default_band_params() -> RankerBandParameterMap:
    """
    This function calculates a set of parameters used by the ranker for each band.
    """
    m2 = {Band.BAND4: 0.0, Band.BAND3: 1.0, Band.BAND2: 6.0, Band.BAND1: 20.0}
    xb = 0.8
    b1 = 1.2

    params = {Band.BAND4: RankerBandParameters(m1=0.00, b1=0.1, m2=0.00, b2=0.0, xb=0.8, xb0=0.0, xc0=0.0)}
    for band in [Band.BAND3, Band.BAND2, Band.BAND1]:
        # Intercept for linear segment.
        b2 = b1 + 5. - m2[band]

        # Parabola coefficient so that the curves meet at xb: y = m1*xb**2 + b1 = m2*xb + b2.
        m1 = (m2[band] * xb + b2) / xb ** 2
        params[band] = RankerBandParameters(m1=m1, b1=b1, m2=m2[band], b2=b2, xb=xb, xb0=0.0, xc0=0.0)

        # Zero point for band separation.
        b1 += m2[band] * 1.0 + b2

    return params
