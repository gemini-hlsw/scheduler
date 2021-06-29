EMPTY = -1
UNSCHEDULABLE = -2

from greedy_max.schedule import *
from typing import List, NoReturn, Optional, Sized, Tuple

class Plan:
    """
    The final plan created by the scheduler that includes the schedule (for both sites) itself
    and the observations for each site.
    """
    def __init__(self, total_time_slots, sites):              
        self.schedule = {Site.GS: np.full(total_time_slots, EMPTY),
                         Site.GN: np.full(total_time_slots, EMPTY)}
        self.units = {Site.GS: [], Site.GN: []}
        self.sites = sites[:]

    def timeslots_not_scheduled(self) -> int:
        """
        Determine the number of empty time slots at both sites.
        """
        return sum(len(self._empty_slots(site)) for site in self.sites)
    
    def _empty_slots(self, site: Site) -> np.ndarray:
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
    
    def schedule_observations(self, site: Site, science: List[Observation], start: int, end: int) -> NoReturn:
        """
        Add all the observations inside a unit in the time slots from start to finish.
        """
        for sci in science:
            if start < end:
                    sci_length = end - start+ 1  if start + sci.length - 1 > end else sci.length
                    self.schedule_observation(site, sci.idx, start, start + sci_length)
                    sci.observed+=sci_length 
                    start+=sci_length

    def is_observation_scheduled(self, site: Site, obs_idx: int) -> bool:
        """
        Determine if an observation is partially or fully scheduled at the specified site.
        """
        return len(self.slots_for_observation(site, obs_idx)) != 0

    def is_complete(self, site: Site) -> bool:
        return len(self._empty_slots(site)) == 0

    def _intervals(self, empty_slots: np.ndarray) -> np.ndarray:
        """
        Given the empty slots in the schedule, returns an array with the indices with the
        intervals that can schedule an observation
        """
        ni = len(empty_slots)
        cvec = np.zeros(ni, dtype=int)
        nint = 1
        cvec[0] = nint
        for j in range(1, ni):
            if empty_slots[j] != (empty_slots[j - 1] + 1):
                nint = nint + 1
            cvec[j] = nint 

        idx = np.digitize(cvec, bins=np.arange(ni) + 1)

        return idx

    def get_earliest_available_interval(self, site: Site) -> np.ndarray:
        """
        Get the earliest available space in the schedule that can allocate an observation 
        """  
        empty_slots =  self._empty_slots(site)
        return empty_slots[np.where(self._intervals(empty_slots) == 1)[0][:]]
    
    def get_observation_orders(self, site: Site) -> List[Tuple[int]]:
        """
        Get the observation idx and position for all the schedule observations in order
        Return
        -------

        orders: List of a 3-dimensional tuple: observation.idx, initial position
                and last position in the plan.
        """
        
        schedule = self.schedule[site]
        obs_comparator = self.schedule[site][0]
        start = 0

        orders = []
        for position ,obs_idx in enumerate(schedule):
            if obs_idx != obs_comparator:
                orders.append((obs_comparator,start,position-1))
                start = position
                obs_comparator = obs_idx
            elif position == len(schedule)-1:
                orders.append((obs_comparator,start,position))
            
        return orders
    
    def get_unit_by_observation(self, site: Site, obs_idx: int ) -> Optional[SchedulingUnit]:
        """
        Retrieve the unit that owns the observation 
        """
        unit = None
        for u in self.units[site]:
            if obs_idx in u:
                unit = u
        
        return unit

    def __getitem__(self, site: Site) -> np.ndarray:
        """
        Get the schedule for the specified site.
        """
        return self.schedule[site]
