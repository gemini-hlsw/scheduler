# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import abstractmethod, ABC
from copy import deepcopy
from typing import Dict, FrozenSet, Tuple

import astropy.units as u
import numpy as np
import numpy.typing as npt
from lucupy.minimodel import (ALL_SITES, Band, Group, NightIndex, NightIndices, Observation, ObservationStatus,
                              Program, Site)
from lucupy.types import ListOrNDArray, MinMax

# from scheduler.core.calculations import Scores, GroupDataMap
from .parameters import RankerBandParameterMap, RankerParameters, default_band_params

__all__ = [
    'Ranker',
]


class Ranker(ABC):
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

        self.band_params = default_band_params() if band_params is None else band_params
        self.params = params
        self.collector = collector
        self.night_indices = night_indices
        self.sites = sites

        # For convenience, for each site, create:
        # 1. An empty observation score array.
        # 2. An empty group scoring array, used to collect the group scores.
        # This allows us to avoid having to store a reference to the Collector in the Ranker.
        self._empty_obs_scores: Dict[Site, Dict[NightIndex, npt.NDArray[float]]] = {}
        self._empty_group_scores: Dict[Site, Dict[NightIndex, npt.NDArray[float]]] = {}
        for site in self.sites:
            night_events = collector.get_night_events(site)

            # Create a full zero score that fits the sites, nights, and time slots for observations.
            self._empty_obs_scores[site] = {night_idx: np.zeros(len(night_events.times[night_idx]), dtype=float)
                                            for night_idx in self.night_indices}

            # Create a full zero score that fits the sites, nights, and time slots for group calculations.
            # As this must collect the subgroups, the dimensions are different from observation scores.
            self._empty_group_scores[site] = {night_idx: np.zeros((0, len(night_events.times[night_idx])), dtype=float)
                                              for night_idx in self.night_indices}

    @abstractmethod
    def _combine_score_terms(self,
                             scale_factor: float,
                             metric: float,
                             vis_frac: float,
                             wha: npt.NDArray[float]) -> npt.NDArray[float]:
        """
        Combine the program metric, the remaining visibility fraction and the hour angle
        weighting into a per-time-slot score for one night, scaled by scale_factor.

        Parameters
            scale_factor: scalar; preimaging * user priority * ongoing * program priority.
            metric:       scalar program metric, i.e. metric_slope(...)[0][0].
            vis_frac:     scalar remaining visibility fraction for this night.
            wha:          per-time-slot hour angle weighting, already clamped at 0.

        Returns
            An array the same length as wha. The caller applies the altitude mask and the
            airmass normalization, so do not apply them here.
        """

    def score_group(self, group: Group, group_data_map):
        """
        Calculate the score of a Group.

        This method returns the results in the form of a list, where each entry represents
        one night as per the night_indices array, with the list entries being numpy arrays
        that contain the scoring for each time slot across the night.
        """
        # Check isinstance instead of is_and_group or is_or_group because otherwise, we get warnings.
        if group.is_and_group():
            return self._score_and_group(group, group_data_map)
        elif group.is_or_group():
            return self._score_or_group(group, group_data_map)
        else:
            raise ValueError('Ranker group scoring can only score groups.')

    def metric_slope(self,
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

    def score_observation(self, program: Program, obs: Observation, night_configurations: dict,
                          night_indices: NightIndices):
        """
        Calculate the scores for an observation for each night for each time slot index.
        These are returned as a list indexed by night index as per the night_indices supplied,
        and the list items are numpy arrays of float for each time slot during the specified night.
        """
        # Scores are indexed by night_idx and contain scores for each time slot.
        # We initialize to all zeros.
        scores = deepcopy(self._empty_obs_scores[obs.site])

        target_info = self.collector.get_target_info(obs.id)
        if target_info is None:
            return scores


        obs_nights = sorted(set(night_indices) & target_info.keys())
        if not obs_nights:
            return scores

        remaining = obs.exec_time() - obs.total_used()
        # GPP supports allocated and used times by band, this should give the same results for OCS
        cplt = (program.total_used(obs.band) + remaining) / program.total_awarded(obs.band)

        metric, metric_s = self.metric_slope(np.array([cplt]),
                                              np.array([obs.band.value]),
                                              np.array([0.8]),
                                              program.thesis)

        # Declination for the base target per night.
        dec = {night_idx: target_info[night_idx].coord.dec for night_idx in obs_nights}

        # Hour angle / airmass
        ha = {night_idx: target_info[night_idx].hourangle for night_idx in obs_nights}
        airmass = {night_idx: target_info[night_idx].airmass for night_idx in obs_nights}

        # Get the latitude associated with the site.
        site_latitude = obs.site.location.lat
        if site_latitude < 0. * u.deg:
            dec_diff = {night_idx: np.abs(site_latitude - np.max(dec[night_idx])) for night_idx in obs_nights}
        else:
            dec_diff = {night_idx: np.abs(np.min(dec[night_idx]) - site_latitude) for night_idx in obs_nights}

        c = {night_idx: self.params.dec_diff_less_40 if angle < 40. * u.deg else self.params.dec_diff
             for night_idx, angle in dec_diff.items()}
        # c = np.array([self.params.dec_diff_less_40 if angle < 40. * u.deg
        #               else self.params.dec_diff for angle in dec_diff])

        # Todo: check units for hour angle are correct!!!
        wha = {night_idx: c[night_idx][0] + c[night_idx][1] * ha[night_idx] / u.hourangle
               + (c[night_idx][2] / u.hourangle ** 2) * ha[night_idx] ** 2
               for night_idx in obs_nights}
        kk = {night_idx: np.where(wha[night_idx] <= 0.)[0] for night_idx in obs_nights}
        for night_idx in obs_nights:
            wha[night_idx][kk[night_idx]] = 0.
        # print(f'   max wha: {np.max(wha[0]):.2f}  visfrac: {target_info[0].rem_visibility_frac:.5f}')

        # Telescope altitude restrictions - set score to 0 if the altitude is outside the limits
        targ_alt = {night_idx: target_info[night_idx].alt for night_idx in obs_nights}
        alt_include = {night_idx: np.ones(len(targ_alt[night_idx])) for night_idx in obs_nights}
        jj = {night_idx: np.where(np.logical_or(targ_alt[night_idx] < self.params.altitude_limits[obs.site][MinMax.MIN],
                                                 targ_alt[night_idx] > self.params.altitude_limits[obs.site][MinMax.MAX]))[0]
              for night_idx in obs_nights}
        for night_idx in obs_nights:
            alt_include[night_idx][jj[night_idx]] = 0.0

        # Scale factor terms: preimaging * user priority * ongoing * prog priority
        # MOS pre-imaging boost
        if obs.preimaging:
            scale_factor = self.params.preimaging_factor
        else:
            scale_factor = 1.0

        # Effective user priority
        # Normalized to 1, use priority_factor to scale
        priority_value = obs.priority.value
        # if obs.status ==  ObservationStatus.ONGOING and obs.total_used() > ZeroTime:
        #     priority_value += 1
        scale_factor *= 1. + (priority_value - program.mean_priority())/self.params.priority_factor

        # If Ongoing give boost
        # if obs.status ==  ObservationStatus.ONGOING and obs.total_used() > ZeroTime:
        if obs.status ==  ObservationStatus.ONGOING:
            scale_factor *= self.params.ongoing_factor

        # Program priority (from calendar, e.g. PV, classical)
        nc = night_configurations[obs.site]
        program = self.collector.get_program(obs.id.program_id())
        prog_priority = {night_idx: self.params.program_priority if nc[night_idx].filter.program_priority_filter_any(program)
                         else 1.0 for night_idx in obs_nights}

        scale_factor *= prog_priority[obs_nights[-1]]

        # Instrument priority, e.g. visitor instrument blocks
        # ToDo: finish this if the visibility calc does not take the instrument calendar into account,
        #  the filters may need updating
        # inst_priority = {night_idx: self.params.program_priority if nc[night_idx].filter.resource_priority_filter(obs)
        #                  else 1.0 for night_idx in obs_nights}
        # scale_factor *= inst_priority[night_idx]

        # Divide by the minimum airmass (mainly for cross-site scoring tests)
        p = {night_idx: self._combine_score_terms(scale_factor,
                                                  metric[0],
                                                  target_info[night_idx].rem_visibility_frac,
                                                  wha[night_idx]) * alt_include[night_idx] /
                        (np.min(airmass[night_idx]) ** self.params.air_power)
             for night_idx in obs_nights}

        # Assign scores in p to all indices where visibility constraints are met.
        # They will otherwise be 0 as originally defined.
        for night_idx in obs_nights:
            slot_indices = target_info[night_idx].visibility_slot_idx
            scores[night_idx].put(slot_indices, p[night_idx][slot_indices])

        return scores

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

        # For each night, calculate the score for the group over its subgroups.
        # This may not be the same as using the observation scoring, since for groups, the score has been adjusted in
        # the Selector for things like wind, conditions matching, etc.

        nights_to_schedule = list(list(group_data_map.values())[0].group_info.scores.keys())
        for night_idx in nights_to_schedule:
            # What we want for the night is a numpy array of size (#obs, #timeslots in night)
            # where the rows are the observation scores. Then we will combine them.
            for unique_group_id in (g.unique_id for g in group.children):
                # To get this, we turn the scores of the children into a (1, #timeslots in night) array to append
                # to the numpy array for the night.
                subgroup_scores = np.array([group_data_map[unique_group_id].group_info.scores[night_idx]])
                scores[night_idx] = np.append(scores[night_idx], subgroup_scores, axis=0)

        # Combine the scores as per the score_combiner and return.
        # apply_along_axis results in a (1, #timeslots in night) array, so we have to take index 0.
        combine_scores = {night_idx: np.apply_along_axis(self.params.score_combiner, 0, scores[night_idx])[0]
                          for night_idx in nights_to_schedule}
        # print(group.id.id, combine_scores[nights_to_schedule[0]])
        return combine_scores

    def _score_or_group(self, group: Group, group_data_map):
        """
        Calculate the scores for each night and time slot of an OR Group.
        TODO: This is TBD and requires more design work.
        TODO: In fact, OcsProgramProvider does not even support OR Groups.
        """
        raise NotImplementedError
