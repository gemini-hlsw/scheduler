# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Mapping, Tuple, final

import astropy.units as u
from astropy.coordinates import Angle
import numpy as np
import numpy.typing as npt
from lucupy.minimodel import ALL_SITES, Group, Band, NightIndices, Observation, Program, Site, Priority
from lucupy.types import ListOrNDArray, MinMax

# from scheduler.core.calculations import Scores, GroupDataMap
from .base import Ranker

__all__ = [
    'DefaultRanker',
    'RankerParameters',
    'RankerBandParameters'
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
    met_power: float = 1.0
    vis_power: float = 1.0
    wha_power: float = 1.0
    program_priority: float = 10.0
    priority_factor: float = 8.0
    preimaging_factor: float = 1.25
    # altitude_limits: Dict[Site, Dict[MinMax, Angle]] = field(default_factory=lambda: _def_alt_limits_site)
    gs_altitude_limits: Dict[MinMax, Angle] = field(default_factory=lambda: _def_alt_limits)
    gn_altitude_limits: Dict[MinMax, Angle] = field(default_factory=lambda: _def_alt_limits)
    altitude_limits: Dict[Site, Dict[MinMax, Angle]] = field(init={})

    # user_priority_factors: Dict[Priority, float] = field(default_factory=lambda: _default_user_priority_factors)

    # Weighted to slightly positive HA.
    dec_diff_less_40: npt.NDArray[float] = field(default_factory=lambda: np.array([3., 0., -0.08]))
    # Weighted to 0 HA if Xmin > 1.3.
    dec_diff: npt.NDArray[float] = field(default_factory=lambda: np.array([3., 0.1, -0.06]))

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


def _default_band_params() -> RankerBandParameterMap:
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


class DefaultRanker(Ranker):
    """
    The Ranker is a scoring algorithm used by the Selector to assign scores
    to Groups. It calculates first all the scores for the observations for
    the given night indices and then stores this information here and uses
    it to agglomerate the scores for a specified Group.
    """

    def __init__(self,
                 collector,  # Creates a circular input if we typehint this.
                 night_indices: NightIndices,
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
                      band: ListOrNDArray[Band],
                      b3min: ListOrNDArray[float],
                      thesis: bool) -> Tuple[npt.NDArray[float], npt.NDArray[float]]:
        """
        Compute the metric and the slope as a function of completeness fraction and band.

        Parameters
            completion: array/list of program completion fractions
            band: integer array of bands for each program
            b3min: array of Band 3 minimum time fractions (Band 3 minimum time / Allocated program time)
            params: dictionary of parameters for the metric
            power: exponent on completion, power=1 is linear, power=2 is parabolic
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
            b2 = 0.0
            if self.params.power == 1:
                b2 = (xb * (self.band_params[curr_band].m1 - self.band_params[curr_band].m2) +
                      self.band_params[curr_band].xb0 + self.band_params[curr_band].b1)
            elif self.params.power == 2:
                b2 = self.band_params[curr_band].b2 + self.band_params[curr_band].xb0 + self.band_params[curr_band].b1

            # Finally, calculate piecewise the metric and slope.
            if completion[idx] <= eps:
                metric[idx] = 0.0
                metric_slope[idx] = 0.0
            elif completion[idx] < xb:
                metric[idx] = (self.band_params[curr_band].m1 * completion[idx] ** self.params.power
                               + self.band_params[curr_band].b1)
                metric_slope[idx] = (self.params.power * self.band_params[curr_band].m1
                                     * completion[idx] ** (self.params.power - 1.0))
            elif completion[idx] < 1.0:
                metric[idx] = self.band_params[curr_band].m2 * completion[idx] + b2
                metric_slope[idx] = self.band_params[curr_band].m2
            else:
                metric[idx] = self.band_params[curr_band].m2 * 1.0 + b2 + self.band_params[curr_band].xc0
                metric_slope[idx] = self.band_params[curr_band].m2

        if thesis:
            metric += self.params.thesis_factor

        return metric, metric_slope

    def score_observation(self, program: Program, obs: Observation, night_configurations: dict):
        """
        Calculate the scores for an observation for each night for each time slot index.
        These are returned as a list indexed by night index as per the night_indices supplied,
        and the list items are numpy arrays of float for each time slot during the specified night.
        """
        # Scores are indexed by night_idx and contain scores for each time slot.
        # We initialize to all zeros.
        scores = deepcopy(self._empty_obs_scores[obs.site])
        metrics = deepcopy(self._empty_metrics[obs.site])

        # target_info is a map from night index to TargetInfo.
        # We require it to proceed for hour angle / elevation information and coordinates.
        # If it is missing, just return scores of 0.
        target_info = self.collector.get_target_info(obs.id)
        if target_info is None:
            return scores

        remaining = obs.exec_time() - obs.total_used()
        # GPP supports allocated and used times by band, this should give the same results for OCS
        cplt = (program.total_used(obs.band) + remaining) / program.total_awarded(obs.band)

        metric, metric_s = self._metric_slope(np.array([cplt]),
                                              np.array([obs.band.value]),
                                              np.array([0.8]),
                                              program.thesis)

        # Declination for the base target per night.
        dec = {night_idx: target_info[night_idx].coord.dec for night_idx in self.night_indices}

        # Hour angle / airmass
        ha = {night_idx: target_info[night_idx].hourangle for night_idx in self.night_indices}

        # Get the latitude associated with the site.
        site_latitude = obs.site.location.lat
        if site_latitude < 0. * u.deg:
            dec_diff = {night_idx: np.abs(site_latitude - np.max(dec[night_idx])) for night_idx in self.night_indices}
        else:
            dec_diff = {night_idx: np.abs(np.min(dec[night_idx]) - site_latitude) for night_idx in self.night_indices}

        c = {night_idx: self.params.dec_diff_less_40 if angle < 40. * u.deg else self.params.dec_diff
             for night_idx, angle in dec_diff.items()}
        # c = np.array([self.params.dec_diff_less_40 if angle < 40. * u.deg
        #               else self.params.dec_diff for angle in dec_diff])

        wha = {night_idx: c[night_idx][0] + c[night_idx][1] * ha[night_idx] / u.hourangle
               + (c[night_idx][2] / u.hourangle ** 2) * ha[night_idx] ** 2
               for night_idx in self.night_indices}
        kk = {night_idx: np.where(wha[night_idx] <= 0.)[0] for night_idx in self.night_indices}
        for night_idx in self.night_indices:
            wha[night_idx][kk[night_idx]] = 0.
        # print(f'   max wha: {np.max(wha[0]):.2f}  visfrac: {target_info[0].rem_visibility_frac:.5f}')

        # Telescope altitude restrictions - set score to 0 if the altitude is outside the limits
        targ_alt = {night_idx: target_info[night_idx].alt for night_idx in self.night_indices}
        alt_include = {night_idx: np.ones(len(targ_alt[night_idx])) for night_idx in self.night_indices}
        jj = {night_idx: np.where(np.logical_or(targ_alt[night_idx] < self.params.altitude_limits[obs.site][MinMax.MIN],
                                                 targ_alt[night_idx] > self.params.altitude_limits[obs.site][MinMax.MAX]))[0]
              for night_idx in self.night_indices}
        for night_idx in self.night_indices:
            alt_include[night_idx][jj[night_idx]] = 0.0

        # MOS pre-imaging boost?
        if obs.preimaging:
            preimaging = self.params.preimaging_factor
        else:
            preimaging = 1.0

        # Effective user priority
        # Normalized to 1, use priority_factor to scale
        user_priority = 1. + (obs.priority.value - program.mean_priority())/self.params.priority_factor

        # Program priority (from calendar)
        nc = night_configurations[obs.site]
        program = self.collector.get_program(obs.id.program_id())
        prog_priority = {night_idx: self.params.program_priority if nc[night_idx].filter.program_priority_filter_any(program)
                         else 1.0 for night_idx in self.night_indices}
        # print(obs.unique_id, night_idx, prog_priority)

        # p = {night_idx: (metric[0] ** self.params.met_power) *
        p = {night_idx: (preimaging * user_priority * prog_priority[night_idx]) * (metric[0] ** self.params.met_power) *
                        (target_info[night_idx].rem_visibility_frac ** self.params.vis_power) *
                        (wha[night_idx] ** self.params.wha_power) * alt_include[night_idx]
             for night_idx in self.night_indices}

        # Assign scores in p to all indices where visibility constraints are met.
        # They will otherwise be 0 as originally defined.
        for night_idx in self.night_indices:
            slot_indices = target_info[night_idx].visibility_slot_idx
            scores[night_idx].put(slot_indices, p[night_idx][slot_indices])
            metrics[night_idx].append(float(metric[0]))
            # print(f'   max score on night {night_idx}: {np.max(scores[night_idx])}')

        return scores, metrics

    # TODO: Should we be considering the scores of the subgroups or the scores of the
    # TODO: observations when calculating the score of this group?
    def _score_and_group(self, group: Group, group_data_map):
        """
        Calculate the scores for each night and time slot of an AND Group.
        """
        # TODO: An AND group could theoretically be at multiple sites if it contained
        # TODO: an OR group, but check before changing the score to be per site as well.
        if len(group.sites()) != 1:
            raise ValueError(f'AND group {group.group_name} has too many sites: {len(group.sites())}')

        # Determine the length of the nights and create an empty score array for each night.
        site = list(group.sites())[0]
        scores = deepcopy(self._empty_group_scores[site])
        metrics = deepcopy(self._empty_metrics[site])

        # For each night, calculate the score for the group over its subgroups.
        # This may not be the same as using the observation scoring, since for groups, the score has been adjusted in
        # the Selector for things like wind, conditions matching, etc.
        for night_idx in self.night_indices:
            # What we want for the night is a numpy array of size (#obs, #timeslots in night)
            # where the rows are the observation scores. Then we will combine them.
            for unique_group_id in (g.unique_id for g in group.children):
                # To get this, we turn the scores of the children into a (1, #timeslots in night) array to append
                # to the numpy array for the night.
                subgroup_scores = np.array([group_data_map[unique_group_id].group_info.scores[night_idx]])
                subgroup_metrics = np.array([group_data_map[unique_group_id].group_info.metrics[night_idx]])
                scores[night_idx] = np.append(scores[night_idx], subgroup_scores, axis=0)
                metrics[night_idx] += [x for sublist in subgroup_metrics for x in sublist]

        # Combine the scores as per the score_combiner and return.
        # apply_along_axis results in a (1, #timeslots in night) array, so we have to take index 0.
        combine_scores = {night_idx: np.apply_along_axis(self.params.score_combiner, 0, scores[night_idx])[0]
                          for night_idx in self.night_indices}
        return combine_scores, metrics

    def _score_or_group(self, group: Group, group_data_map):
        raise NotImplementedError
