from dataclasses import dataclass
from enum import Enum, unique
import numpy as np
from astropy.time import Time
from astropy.units.quantity import Quantity
from typing import Dict, List, Sized


@unique
class Site(Enum):
    """
    The sites (telescopes) available to an observation.
    """
    GS = 'gs'
    GN = 'gn'


EMPTY = -1
UNSCHEDULABLE = -2


class Plan:
    """
    The final plan created by the scheduler that includes the schedule (for both sites) itself
    and the observations for each site.
    """
    def __init__(self, total_time_slots, sites):              
        self.schedule = {Site.GS: np.full(total_time_slots, EMPTY),
                         Site.GN: np.full(total_time_slots, EMPTY)}
        self.observations = None
        self.sites = sites[:]

    def timeslots_not_scheduled(self) -> int:
        """
        Determine the number of empty time slots at both sites.
        """
        return sum(len(self.empty_slots(site)) for site in self.sites)
    
    def empty_slots(self, site: Site) -> np.ndarray:
        """
        Determine the number of empty time slots at a given site..
        """
        return np.where(self.schedule[site] == EMPTY)[0][:]

    def slots_for_observation(self, site: Site, obs_idx: int = EMPTY) -> Sized:
        """
        Determine an array of the time slots for a given observation by index at a given site.
        """
        return np.where(self.schedule[site] == obs_idx)[0][:]

    def schedule_observation(self, site: Site, obs_idx: int, start: int, finish: int) -> None:
        """
        In the plan for the specified site, add the observation obs_idx in the time slots
        from start to finish.
        """
        self.schedule[site][start:finish] = obs_idx

    def is_observation_scheduled(self, site: Site, obs_idx: int) -> bool:
        """
        Determine if an observation is partially or fully scheduled at the specified site.
        """
        return len(self.slots_for_observation(site, obs_idx)) != 0

    def __getitem__(self, site: Site) -> np.ndarray:
        """
        Get the schedule for the specified site.
        """
        return self.schedule[site]


@dataclass
class TimeWindow:
    """
    A single time window that defines the boundary where an observation can be scheduled.
    Contains also the calibration time, total observation time and the time intervals in the schedule
    for that observation.
    """
    start: int
    end: int
    nobswin: int
    total_time: int
    calibration_time: int
    intervals: Dict[Site, np.ndarray]


def sites_from_column_names(colnames: List[str], column: str = 'weight') -> List[Site]:
    """
    Given a column name prefix (e.g. weight), iterate through the column names provided,
    finding column names of the form weight_gs / weight_gn, and return a list of the sites
    found.
    """
    return [Site(name[name.rfind('_')+1:]) for name in colnames if column in name]


def get_order(plan):
    """
    For observations scheduled in plan, get the observation indices in the order they appear, and return the
    array indices of schedule period boundaries observation.

    Example
    -------
    >>> plan = [2, 2, 2, 2, 1, 1, 1, 1, 5, 5, 4, 4, 4, 4]
    >>> index_order, index_start, index_end = get_order(plan)
    >>> print(index_order)
    >>> print(index_start)
    >>> print(index_end)
    [2, 1, 5, 4]
    [0, 4, 8, 10]
    [3, 7, 9, 13]

    Parameters
    ----------
    plan : numpy integer array
        Observation indices throughout night.

    Returns
    -------
    order : list of ints
        order that indices appear in plan.

    i_start : list of ints
        indices of time block beginnings corresponding to plan.

    i_end : list of ints
        indices of time block endings corresponding to plan.
    """

    ind_order = [plan[0]]
    i_start = [0]
    i_end = []
    for i in range(1, len(plan)):
        prev = plan[i - 1]
        if plan[i] != prev:
            ind_order.append(plan[i])
            i_end.append(i - 1)
            i_start.append(i)
        if i == len(plan) - 1:
            i_end.append(i)
    return ind_order, i_start, i_end


def short_observation_id(obsid):
    """
    Return the short form of the observation ID.
    """
    idsp = obsid.split('-')
    return idsp[0][1] + idsp[1][2:5] + '-' + idsp[2] + '-' + idsp[3] + '[' + idsp[4] + ']'


# TODO: What is dt / differential total time length?
def time_slot_length(time_strings: np.ndarray) -> Quantity:
    """
    Get dt

    Parameters
    ----------
    time_strings : array of strings
        iso format strings of utc or local times in timetable

    Returns
    -------
    dt : '~astropy.units.quantity.Quantity'
        differential tot_time length
    """
    return (Time(time_strings[1]) - Time(time_strings[0])).to('hour').round(7)
