from typing import List, Dict
from greedy_max.schedule import Observation, Visit
from common.structures.site import Site
from astropy.units.quantity import Quantity
from astropy.time import Time
from numpy import ndarray

def sites_from_column_names(colnames: List[str], column: str = 'weight') -> List[Site]:
    """
    Given a column name prefix (e.g. weight), iterate through the column names provided,
    finding column names of the form weight_gs / weight_gn, and return a list of the sites
    found.
    """
    return [Site(name[name.rfind('_') + 1:]) for name in colnames if column in name]

def get_observations(visits: List[Visit]) -> Dict[int,Observation]:
    obs = {}
    for visit in visits:
        aux = visit.get_observations() 
        obs = {**obs, **aux} # merge obs and aux 
    return obs

# TODO: What is dt / differential total time length?
def time_slot_length(time_strings: ndarray) -> Quantity:
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

# NOTE: The following methods are legacy code for the old output. This will be address in a different PR (See SCHED-27)
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
