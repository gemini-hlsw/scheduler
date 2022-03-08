from dataclasses import dataclass
import numpy as np

from common.helpers import flatten
from common.minimodel import *
from common.scheduler import SchedulerComponent
from components.collector import Collector, NightIndex


# @dataclass
# class Visit:
#     idx: int
#     site: Site
#     group: Group
#
#     # Standards time in time slots.
#     standard_time: int
#
#     def __post_init__(self):
#         self.score = None
#
#         # Create a Conditions object that uses the most restrictive values over all
#         # observations.
#         self.observations = self.group.observations()
#
#         # TODO: Test this.
#         self.conditions = Conditions(
#             cc=CloudCover(min(flatten((obs.constraints.conditions.cc for obs in self.observations)))),
#             iq=ImageQuality(min(flatten((obs.constraints.conditions.iq for obs in self.observations)))),
#             sb=SkyBackground(min(flatten((obs.constraints.conditions.sb for obs in self.observations)))),
#             wv=WaterVapor(min(flatten((obs.constraints.conditions.wv for obs in self.observations))))
#         )
#
#     def length(self) -> int:
#         """
#         Calculate the length of the unit based on both observation and calibrations times
#         """
#         # length here is acq + time and is a float?
#         obs_slots = sum([obs.length for obs in self.observations])
#
#         if self.standard_time > 0:  # not a single science observation
#             standards_needed = max(1, int(obs_slots // self.standard_time))
#
#             if standards_needed == 1:
#                 cal_slots = self.calibrations[0].length  # take just one
#             else:
#                 cal_slots = sum([cal.length for cal in self.calibrations])
#
#             return obs_slots + cal_slots
#         else:
#             return obs_slots
#
#     def observed(self) -> int:
#         """
#         Calculate the observed time for both observation and calibrations
#         """
#         # observed: observation time on time slots
#         obs_slots = sum([obs.observed for obs in self.observations])
#         cal_slots = sum([cal.observed for cal in self.calibrations])
#         return obs_slots + cal_slots
#
#     def acquisition(self) -> None:
#         """
#         Add acquisition overhead to the total length of each observation in the unit
#         """
#         for observation in self.observations:
#             if observation.observed < observation.length:  # not complete observation
#                 observation.length += observation.acquisition
#
#     def get_observations(self) -> Dict[int, Observation]:
#         total_obs = {}
#         for obs in self.observations:
#             total_obs[obs.idx] = obs
#         for cal in self.calibrations:
#             total_obs[cal.idx] = cal
#         return total_obs
#
#     def airmass(self, obs_idx: int) -> float:
#         """
#         Get airmass values for the observation
#         """
#         if obs_idx in self.observations:
#             return self.observations[obs_idx].visibility.airmass
#         if obs_idx in self.calibrations:
#             return self.calibrations[obs_idx].visibility.airmass
#         else:
#             return None
#
#     def __contains__(self, obs_idx: int) -> bool:
#
#         if obs_idx in [sci.idx for sci in self.observations]:
#             return True
#         elif obs_idx in [cal.idx for cal in self.calibrations]:
#             return True
#         else:
#             return False
#
#     def __str__(self) -> str:
#         return f'Visit {self.idx} \n ' + \
#                f'-- observations: \n {[str(obs) for obs in self.observations]} \n' + \
#                f'-- calibrations: {[str(cal) for cal in self.calibrations]} \n'


@dataclass
class Selector(SchedulerComponent):
    """
    This is the Selector portion of the automated Scheduler.
    It selects the scheduling candidates that are viable for the data collected by
    the Collector.
    """
    collector: Collector

    def __post_init__(self):
        """
        Initialize internal non-input data members.
        """
        ...

    def select(self,
               site: Site):
        """
        Perform the selection of the observations and groups based on:
        * Resource availability
        * 80% chance of completion (TBD)
        """
        ...

    @staticmethod
    def _check_resource_availability(required_resources: Set[Resource],
                                     site: Site,
                                     night_idx: NightIndex) -> bool:
        """
        Determine if the required resources as listed are available at
        the specified site during the given time_period, and if so, at what
        intervals in the time period.
        """
        available_resources = Collector.available_resources(site, night_idx)
        return required_resources.issubset(available_resources)

    @staticmethod
    def _match_conditions(required_conditions: Conditions,
                          actual_conditions: Conditions,
                          neg_ha: bool,
                          too_status: TooType) -> npt.NDArray[float]:
        """
        Determine if the required conditions are satisfied by the actual conditions.
        """
        # TODO: Can we move part of this to the mini-model? Do we want to?

        # Convert the actual conditions to arrays.
        actual_iq = np.asarray(actual_conditions.iq.value)
        actual_cc = np.asarray(actual_conditions.cc.value)
        actual_wv = np.asarray(actual_conditions.wv.value)

        # The mini-model manages the IQ, CC, WV, and SB being the same types and sizes
        # so this check is a bit extraneous: if one has an ndim of 0, they all will.
        scalar_input = actual_iq.ndim == 0 or actual_cc.ndim == 0 or actual_wv.ndim == 0
        if actual_iq.ndim == 0:
            actual_iq = actual_iq[None]
        if actual_cc.ndim == 0:
            actual_cc = actual_cc[None]
        if actual_wv.ndim == 0:
            actual_wv = actual_wv[None]

        # Again, all lengths are guaranteed to be the same by the mini-model.
        length = len(actual_iq)

        # We must be working with required_conditions that are either scalars or the same
        # length as the actual_conditions.
        if len(required_conditions) not in {1, length}:
            raise ValueError(f'Cannot match conditions between arrays of different sizes: {actual_conditions} '
                             f'and {required_conditions}')

        # Determine the positions where the actual conditions are worse than the requirements.
        bad_iq = actual_iq > required_conditions.iq
        bad_cc = actual_cc > required_conditions.cc
        bad_wv = actual_wv > required_conditions.wv

        bad_cond_idx = np.where(np.logical_or(np.logical_or(bad_iq, bad_cc), bad_wv))[0]
        cmatch = np.ones(length)
        cmatch[bad_cond_idx] = 0

        # Penalize for using IQ / CCthat is too good:
        # Multiply the weights by actual value / value where value is better than required and target
        # does not set soon and is not a rapid ToO.
        # TODO: What about interrupting ToOs?

        # This should work as we are adjusting structures that are passed by reference.
        def adjuster(array, value):
            better_idx = np.where(array < value)[0]
            if len(better_idx) > 0 and neg_ha and too_status != TooType.RAPID:
                cmatch[better_idx] = cmatch[better_idx] * array / value
        adjuster(actual_iq, required_conditions.iq)
        adjuster(actual_cc, required_conditions.cc)

        if scalar_input:
            cmatch = np.squeeze(cmatch)
        return cmatch
