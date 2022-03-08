import astropy.units as u
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import List, Mapping, Tuple, Union

from common.minimodel import *


@dataclass(frozen=True, unsafe_hash=True)
class RankerParameters:
    m1: float
    b1: float
    m2: float
    b2: float
    xb: float
    xb0: float
    xc0: float
    

class Ranker:
    @staticmethod
    def score():
        ...

    @staticmethod
    def _params() -> Mapping[Band, RankerParameters]:
        m2 = {Band.BAND4: 0.0, Band.BAND3: 1.0, Band.BAND2: 6.0, Band.BAND1: 20.0}
        xb = 0.8
        b1 = 1.2

        params = {Band.BAND4: RankerParameters(m1=0.00, b1=0.1, m2=0.00, b2=0.0, xb=0.8, xb0=0.0, xc0=0.0)}
        for band in {Band.BAND3, Band.BAND2, Band.BAND1}:
            # Intercept for linear segment.
            b2 = b1 + 5. - m2[band]

            # Parabola coefficient so that the curves meet at xb: y = m1*xb**2 + b1 = m2*xb + b2.
            m1 = (m2[band] * xb + b2) / xb**2
            params[band] = RankerParameters(m1=m1, b1=b1, m2=m2[band], b2=b2, xb=xb, xb0=0.0, xc0=0.0)

            # Zero point for band separation.
            b1 += m2[band]*1.0 + b2

        return params

    @staticmethod
    def _metric_slope(completion: Union[List[float], npt.NDArray[float]],
                      band: npt.NDArray[Band],
                      b3min: npt.NDArray[float],
                      params: Mapping[Band, RankerParameters],
                      comp_exp: int = 1,
                      thesis: bool = False,
                      thesis_factor: float = 0.0) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:

        """
        Compute the metric and the slope as a function of completeness fraction and band.

        Parameters
            completion: array/list of program completion fractions
            band: integer array of bands for each program
            b3min: array of Band 3 minimum time fractions (Band 3 minimum time / Allocated program time)
            params: dictionary of parameters for the metric
            comp_exp: exponent on completion, comp_exp=1 is linear, comp_exp=2 is parabolic
        """
        # TODO: Add error checking to make sure arrays are the appropriate lengths?
        if len(band) != len(completion):
            raise ValueError(f'Incompatible lengths (band={len(band)}, completion={len(completion)}) between band '
                             f'{band} and completion {completion} arrays')

        eps = 1.e-7
        completion = np.asarray(completion)
        nn = len(completion)
        metric = np.zeros(nn)
        metric_slope = np.zeros(nn)

        for idx, curr_band in enumerate(band):
            # If Band 3, then the Band 3 min fraction is used for xb
            if curr_band == Band.BAND3:
                xb = b3min[idx]
                # b2 = xb * (params[curr_band].m1 - params[curr_band].m2) + params[curr_band].xb0
            else:
                xb = params[curr_band].xb
                # b2 = params[curr_band].b2

            # Determine the intercept for the second piece (b2) so that the functions are continuous
            b2 = 0
            if pow == 1:
                b2 = xb * (params[curr_band].m1 - params[curr_band].m2) + params[curr_band].xb0 + params[curr_band].b1
            elif pow == 2:
                b2 = params[curr_band].b2 + params[curr_band].xb0 + params[curr_band].b1

            # Finally, calculate piecewise the metric and slope.
            if completion[idx] <= eps:
                metric[idx] = 0.0
                metric_slope[idx] = 0.0
            elif completion[idx] < xb:
                metric[idx] = params[curr_band].m1 * completion[idx] ** comp_exp + params[curr_band].b1
                metric_slope[idx] = comp_exp * params[curr_band].m1 * completion[idx] ** (comp_exp - 1.0)
            elif completion[idx] < 1.0:
                metric[idx] = params[curr_band].m2 * completion[idx] + b2
                metric_slope[idx] = params[curr_band].m2
            else:
                metric[idx] = params[curr_band].m2 * 1.0 + b2 + params[curr_band].xc0
                metric_slope[idx] = params[curr_band].m2

        if thesis:
            metric += thesis_factor
            # metric *= thesis_factor
            # metric_slope *= thesis_factor

        return metric, metric_slope
