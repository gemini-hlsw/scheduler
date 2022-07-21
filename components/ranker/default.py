from .base import Ranker
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, FrozenSet, Mapping, Tuple

import astropy.units as u
import numpy as np
import numpy.typing as npt

from common.calculations import Scores
from common.minimodel import ALL_SITES, AndGroup, Band, NightIndex, Observation, Program, Site
from common.types import ListOrNDArray
from components.collector import Collector


def _default_score_combiner(x: npt.NDArray[float]) -> npt.NDArray[float]:
    """
    The default function used to combine scores for Groups.
    """
    return np.array([np.max(x)]) if 0 not in x else np.array([0])


@dataclass(frozen=True, unsafe_hash=True)
class RankerParameters:
    """
    Global parameters for the Ranker.
    """
    thesis_factor = 1.1
    power: int = 2
    met_power: float = 1.0
    vis_power: float = 1.0
    wha_power: float = 1.0
    comp_exp: int = 1

    # Weighted to slightly positive HA.
    dec_diff_less_40: npt.NDArray[float] = np.array([3., 0., -0.08])
    # Weighted to 0 HA if Xmin > 1.3.
    dec_diff: npt.NDArray[float] = np.array([3., 0.1, -0.06])

    score_combiner: Callable[[npt.NDArray[float]], npt.NDArray[float]] = _default_score_combiner


@dataclass(frozen=True, unsafe_hash=True)
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


def _default_band_params() -> RankerBandParameterMap:
    """
    This function calculates a set of parameters used by the ranker for each band.
    """
    m2 = {Band.BAND4: 0.0, Band.BAND3: 1.0, Band.BAND2: 6.0, Band.BAND1: 20.0}
    xb = 0.8
    b1 = 1.2

    params = {Band.BAND4: RankerBandParameters(m1=0.00, b1=0.1, m2=0.00, b2=0.0, xb=0.8, xb0=0.0, xc0=0.0)}
    for band in {Band.BAND3, Band.BAND2, Band.BAND1}:
        # Intercept for linear segment.
        b2 = b1 + 5. - m2[band]

        # Parabola coefficient so that the curves meet at xb: y = m1*xb**2 + b1 = m2*xb + b2.
        m1 = (m2[band] * xb + b2) / xb ** 2
        params[band] = RankerBandParameters(m1=m1, b1=b1, m2=m2[band], b2=b2, xb=xb, xb0=0.0, xc0=0.0)

        # Zero point for band separation.
        b1 += m2[band] * 1.0 + b2

    return params


class DefaultRanker(Ranker):
    """
    The Ranker is a scoring algorithm used by the Selector to assign scores
    to Groups. It calculates first all the scores for the observations for
    the given night indices and then stores this information here and uses
    it to agglomerate the scores for a specified Group.
    """

    def __init__(self,
                 collector: Collector,
                 night_indices: npt.NDArray[NightIndex],
                 sites: FrozenSet[Site] = ALL_SITES,
                 params: RankerParameters = RankerParameters(),
                 band_params: RankerBandParameterMap = None):
        """
        We only want to calculate the parameters once since they do not change.
        """
        
        if band_params is None:
            self.band_params = _default_band_params()
        self.params = params
        super().__init__(collector, night_indices, sites)
        
    def _metric_slope(self,
                      completion: ListOrNDArray[float],
                      band: npt.NDArray[Band],
                      b3min: npt.NDArray[float],
                      thesis: bool) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
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
                xb = self.band_params[curr_band].xb
                # b2 = params[curr_band].b2

            # Determine the intercept for the second piece (b2) so that the functions are continuous
            b2 = 0
            if pow == 1:
                b2 = (xb * (self.band_params[curr_band].m1 - self.band_params[curr_band].m2) +
                      self.band_params[curr_band].xb0 + self.band_params[curr_band].b1)
            elif pow == 2:
                b2 = self.band_params[curr_band].b2 + self.band_params[curr_band].xb0 + self.band_params[curr_band].b1

            # Finally, calculate piecewise the metric and slope.
            if completion[idx] <= eps:
                metric[idx] = 0.0
                metric_slope[idx] = 0.0
            elif completion[idx] < xb:
                metric[idx] = (self.band_params[curr_band].m1 * completion[idx] ** self.params.comp_exp
                               + self.band_params[curr_band].b1)
                metric_slope[idx] = (self.params.comp_exp * self.band_params[curr_band].m1
                                     * completion[idx] ** (self.params.comp_exp - 1.0))
            elif completion[idx] < 1.0:
                metric[idx] = self.band_params[curr_band].m2 * completion[idx] + b2
                metric_slope[idx] = self.band_params[curr_band].m2
            else:
                metric[idx] = self.band_params[curr_band].m2 * 1.0 + b2 + self.band_params[curr_band].xc0
                metric_slope[idx] = self.band_params[curr_band].m2

        if thesis:
            metric += self.params.thesis_factor

        return metric, metric_slope

    def _score_obs(self, program: Program, obs: Observation) -> Scores:
        """
        Calculate the scores for an observation for each night for each time slot index.
        These are returned as a list indexed by night index as per the night_indices supplied,
        and the list items are numpy arrays of float for each time slot during the specified night.
        """
        # Scores are indexed by night_idx and contain scores for each time slot.
        # We initialize to all zeros.
        scores = deepcopy(self._zero_scores[obs.site])

        # target_info is a map from night index to TargetInfo.
        # We require it to proceed for hour angle / elevation information and coordinates.
        # If it is missing, just return scores of 0.
        target_info = Collector.get_target_info(obs.id)
        if target_info is None:
            return scores

        # TODO: Check this for correctness in mapping from old calcs to new.
        remaining = obs.exec_time() - obs.total_used()
        cplt = (program.total_used() + remaining) / program.total_awarded()

        metric, metric_s = self._metric_slope(np.array([cplt]),
                                              np.ones(1, dtype=int) * program.band,
                                              np.ones(1) * 0.8,
                                              program.thesis)

        # Declination for the base target per night.
        dec = [target_info[night_idx].coord.dec for night_idx in self.night_indices]

        # Hour angle / airmass
        ha = [target_info[night_idx].hourangle for night_idx in self.night_indices]

        # Get the latitude associated with the site.
        site_latitude = obs.site.location.lat
        if site_latitude < 0. * u.deg:
            dec_diff = [np.abs(site_latitude - np.max(dec[night_idx])) for night_idx in self.night_indices]
        else:
            dec_diff = [np.abs(np.min(dec[night_idx]) - site_latitude) for night_idx in self.night_indices]

        c = np.array([self.params.dec_diff_less_40 if angle < 40. * u.deg
                      else self.params.dec_diff for angle in dec_diff])

        wha = [c[night_idx][0] + c[night_idx][1] * ha[night_idx] / u.hourangle
               + (c[night_idx][2] / u.hourangle ** 2) * ha[night_idx] ** 2
               for night_idx in self.night_indices]
        kk = [np.where(wha[night_idx] <= 0.)[0] for night_idx in self.night_indices]
        for night_idx in self.night_indices:
            wha[night_idx][kk[night_idx]] = 0.

        p = [(metric[0] ** self.params.met_power) *
             (target_info[night_idx].rem_visibility_frac ** self.params.vis_power) *
             (wha[night_idx] ** self.params.wha_power)
             for night_idx in self.night_indices]

        # Assign scores in p to all indices where visibility constraints are met.
        # They will otherwise be 0 as originally defined.
        for night_idx in self.night_indices:
            slot_indices = target_info[night_idx].visibility_slot_idx
            scores[night_idx].put(slot_indices, p[night_idx][slot_indices])

        return scores
    
    # TODO: Should we be considering the scores of the subgroups or the scores of the
    # TODO: observations when calculating the score of this group?
    def _score_and_group(self, group: AndGroup) -> Scores:
        """
        Calculate the scores for each night and time slot of an AND Group.
        """
        # TODO: An AND group could theoretically be at multiple sites if it contained
        # TODO: an OR group, but check before changing the score to be per site as well.
        if len(group.sites()) != 1:
            raise ValueError(f'AND group {group.group_name} has too many sites: {len(group.sites())}')
        night_events = self.collector.get_night_events(list(group.sites())[0])
        group_scores = [np.empty((0, len(night_events.times[night_idx])), dtype=float)
                        for night_idx in self.night_indices]

        # For each night, calculate the score for the group over its observations.
        for night_idx in self.night_indices:
            for obs in group.observations():
                obs_score = self.get_observation_scores(obs.id)[night_idx]
                group_scores[night_idx] = np.append(group_scores[night_idx], np.array([obs_score]), axis=0)

        # Combine the scores as per the score_combiner and return.
        return [np.apply_along_axis(self.params.score_combiner, 0, group_scores[night_idx])[0]
                for night_idx in self.night_indices]
