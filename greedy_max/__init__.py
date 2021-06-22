from copy import copy
from dataclasses import dataclass
from hashlib import new
from astropy.time import Time
from typing import Deque, Dict, List, NoReturn, Optional, Sized, Tuple

from numpy.core.numeric import indices
from greedy_max.schedule import *
import astropy.units as u
from astropy.units.quantity import Quantity

from tabulate import tabulate
from collections import deque
from copy import deepcopy

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
        return len(self._empty_slots(site)) < 0

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

    def get_available_interval(self, site: Site) -> np.ndarray:
        """
        Get the first available space in the schedule that can allocate an observation 
        """  
        empty_slots =  self._empty_slots(site)
        return empty_slots[np.where(self._intervals(empty_slots) == 1)[0][:]]
    
    def get_observation_orders(self, site: Site) -> List[Tuple[int]]:
        
        schedule = self.schedule[site]
        obs_cmp = self.schedule[site][0]
        start = 0

        orders = []
        for position ,obs_idx in enumerate(schedule):
            if obs_idx != obs_cmp:
                orders.append((obs_cmp,start,position-1))
                start = position
                obs_cmp = obs_idx
            elif position == len(schedule)-1:
                orders.append((obs_cmp,start,position))
            
        return orders
    
    def get_unit_by_observation(self, site: Site, obs_idx: int ) -> Optional[SchedulingUnit]:
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

@dataclass
class TimeWindow:
    """
    A single time window that defines the boundary where an observation can be scheduled.
    Contains also the calibration time, total observation time and the time intervals in the schedule
    for that observation.
    """
    start: int
    end: int
    length: int
    time_slots: int
    indices: np.ndarray
    min_slot_time: int
    intervals: Dict[Site, np.ndarray]


def sites_from_column_names(colnames: List[str], column: str = 'weight') -> List[Site]:
    """
    Given a column name prefix (e.g. weight), iterate through the column names provided,
    finding column names of the form weight_gs / weight_gn, and return a list of the sites
    found.
    """
    return [Site(name[name.rfind('_') + 1:]) for name in colnames if column in name]

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

def get_observations(units: List[SchedulingUnit]) -> Dict[int,Observation]:
    obs = {}
    for unit in units:
        aux = unit.get_observations() 
        obs = {**obs, **aux}
    return obs

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


class GreedyMax:
    def __init__(self, obs: List[SchedulingUnit], time_slots: TimeSlots, sites: List[Site],
                 tmin: Quantity = 30.0 * u.min, verbose: bool = False):
        self.plan = Plan(time_slots.total, sites)
        self.observations = obs
        self.time_slots = time_slots
        self.tmin = tmin
        self.sites = sites
        self.verbose = verbose
        #self.min_slot_time = int(np.ceil(tmin.to(u.h) / time_slots.slot_length.to(u.h)))

    def min_slot_time(self):
        return int(np.ceil(self.tmin.to(u.h) / self.time_slots.slot_length.to(u.h)))

    def _match_airmass(self, site: Site, start: int, end: int, 
                       science_obs: List[Observation], 
                       calibrations: List[Observation]) -> Tuple[Observation, bool]:

        airmass = self.time_slots.airmass[site]
        min_delta = 99999.
        for calibration in calibrations:
            

            # Try std before
            new_start = start
            xmean_std = np.mean(airmass[calibration.idx][new_start:new_start + calibration.length])
            new_start += calibration.length
            
            sci_airmass = np.zeros(len(science_obs))

            for sci_position, science in enumerate(science_obs):
                if new_start < end:
                    science_length = science.length
                    if new_start + science_length - 1 > end:
                        science_length = end - new_start+1
                    sci_airmass[sci_position] = np.mean(airmass[science.idx][new_start:new_start+science_length])
                    new_start+=science_length
            
            xmean_sci = np.mean(sci_airmass)
            delta_before = np.abs(xmean_std - xmean_sci)

            # Try std after
            new_end = end - calibration.length + 1
            xmean_std = np.mean(airmass[calibration.idx][new_end:new_end+calibration.length])
            new_start = start

            sci_airmass = np.zeros(len(science_obs))
            for sci_position, science in enumerate(science_obs):
                if new_start < new_end:
                    science_length = science.length
                    if new_start + science_length - 1 > new_end:
                        science_length = new_end - new_start + 1
                    sci_airmass[sci_position] = np.mean(airmass[science.idx][new_start:new_start+science_length])
                    new_start+=science_length

            xmean_sci = np.mean(sci_airmass)
            delta_after = np.abs(xmean_std - xmean_sci)

            # Compare airmass differences
            if delta_before <= delta_after:
                delta = delta_before
                placement = True # before
            else:
                delta = delta_after
                placement = False # after       

            if delta < min_delta:
                min_delta = delta
                best_standard = calibration
                best_placement = placement
        
        return best_standard, best_placement

    def _find_max_observation(self) -> Tuple[
                                            Optional[SchedulingUnit],
                                            Optional[TimeWindow]
                                        ]:
        """
        Select the observation with the maximum weight in a time interval

        Returns
        -------
        max_obs : Observation object (or None)

        time_window : TimeWindow object (or None)

        """
        max_weight = 0.  # maximum weight in time interval
        intervals = {}  # save intervals for each site for later use

        max_obs = None
        time_window = TimeWindow(0, 0, 0, 0, None, 0, None) 
        min_slot_time = self.min_slot_time()
        
        for site in self.sites:    
            if not self.plan.is_complete(site):
    
                first_interval = self.plan.get_available_interval(site)
                intervals[site] = first_interval

                if self.verbose:
                    print('iint: ' ,first_interval)
                    print('site: ', site)

                for observation in self.observations:
                    obs_idx = observation.idx
                    # Get the maximum weight in the interval.
                    wmax = self.time_slots.max_weight(site, obs_idx, first_interval)

                    if wmax > max_weight:
                        
                        candidate_intervals = self.time_slots.non_zero_intervals(site, obs_idx, first_interval)

                        time_slots_needed = observation.length() - observation.observed()


                        used_min_slot = time_slots_needed if time_slots_needed - min_slot_time <= min_slot_time \
                                                          else min_slot_time

                        max_weight_on_interval = 0.0
                        max_interval = None
                        for interval in candidate_intervals:
                            interval_length = interval[1]-interval[0]
                            
                            integral_max_weight = self.time_slots.max_weight(site, obs_idx, 
                                                                    first_interval[interval[0]:interval[1]])
                            # The length of the non-zero interval must be at least as larget as
                            # the minimum length
                            if integral_max_weight > max_weight_on_interval \
                               and interval_length >= used_min_slot:
                                max_weight_on_interval = integral_max_weight
                                max_interval = interval
                        
                        # The select 
                        if max_weight_on_interval > max_weight and (time_slots_needed <= interval_length 
                                                                    or observation.can_be_split):
                            max_weight = max_weight_on_interval
                            # Boundaries of available window
                            time_window.start = first_interval[max_interval[0]]
                            time_window.end = first_interval[max_interval[1]]
                            time_window.indices = first_interval[time_window.start:time_window.end+1]
                            time_window.length = time_window.end -time_window.start + 1
                            time_window.time_slots= time_slots_needed
                            observation.site = site
                            max_obs = observation
                           
                            if self.verbose:
                                print('maxweight', max_weight)
                                print('max obs: ', observation)
                                print('iimax', observation.idx)
                                print('start',  time_window.start)
                                print('end', time_window.end)
                                print('obs idxs', observation.observations[0])
                                print('smax', observation.site)
                               
                        else:
                            self.time_slots.weights[site][obs_idx][intervals[site]] = 0.

        time_window.intervals = intervals
        return max_obs, time_window

    def _insert(self, max_observation: Optional[SchedulingUnit], time_window: Optional[TimeWindow]) -> bool:
        """
        Insert an observation to the final plan, trying to shift the observation inside the time window.

        Parameters
        ----------
        max_observation : Observation

        time_window : TimeWindow

        Returns
        -------
        True if the observation is inserted in the plan, otherwise False

        """

        if not max_observation:
            if self.verbose:
                print('Unscheduble')
            for site in self.sites:
                if len(time_window.intervals[site]) > 0:
                    self.plan.schedule[site][time_window.intervals[site]] = UNSCHEDULABLE
            return False

        # Place observation within available window --
        print(max_observation)
        max_site = max_observation.site
        max_idx = max_observation.idx
        max_weights = self.time_slots.weights[max_site][max_idx]


        if time_window.time_slots > 0  and time_window.time_slots <= time_window.length:
            if not self.plan.is_observation_scheduled(max_site, max_idx):
                # Determine schedule placement for maximum integrated weight
                max_integral_weight = 0.0

                #TODO: ToO case not added yet because there is no data to test it yet
                # Schedule interrupt ToO at beginning of window
                #if max_observation.priority == 'interrupt': 
                #    start = time_window.start
                #    end = time_window.start + time_window.time_slots - 1

                # NOTE: I'm not clear on this documentation.
                if time_window.time_slots > 1:
                    # NOTE: integrates over one extra time slot...
                    # ie. if nttime = 14, then the program will choose 15
                    # x values to do trapz integration (therefore integrating
                    # 14 time slots).
                    if self.verbose:
                        print('\nIntegrating max obs. over window...')
                        print('wstart', time_window.start)
                        print('wend', time_window.end)
                        print('nttime', time_window.time_slots)
                        print('j values', np.arange(time_window.start, time_window.end - time_window.time_slots + 2))

                    # Determine placement for maximum integrated weight 
                    for window_idx in range(time_window.start, time_window.end - time_window.time_slots + 2):
                        
                        integral_weight = sum(max_weights[window_idx: window_idx + time_window.time_slots])

                        #if self.verbose:
                        #    NOTE: ST, 22Jun21 - This verbose is commented because it adds too much info on screen
                        #    only uncomment if necessary.
                        #    print('j range', window_idx, window_idx + time_window.time_slots - 1)
                        #    print('obs weight', max_weights[window_idx:window_idx + time_window.time_slots])
                        #    print('integral', integral_weight)
                        
                        if integral_weight > max_integral_weight:
                            max_integral_weight = integral_weight
                            start = window_idx
                            end = start + time_window.time_slots - 1
                else:
                    start = np.argmax(max_weights[indices])
                    max_integral_weight = np.amax(max_weights[start])
                    end = start + time_window.time_slots - 1

                if self.verbose:
                    print('max integral of weight func (maxf)',  max_integral_weight)
                    print('index start', start)
                    print('index end', end)

                # Shift to start or end of night if within minimum block time from boundary.
                # NOTE: BM, 2021may13 - I believe the following code related to the window
                # boundary does the same thing, so the start/end night check seems redundant
                # Nudge:
                #if start < self.min_slot_time:
                #    if self.plan[max_site][0] == -1 and max_weights[0] > 0:
                #        start = 0
                #        end = start + time_window.total_time - 1
                #elif self.time_slots.total - end < self.min_slot_time:
                #    if self.plan[max_site][-1] == -1 and max_weights[-1] > 0:
                #        end = self.time_slots.total - 1
                #        start = end - time_window.total_time + 1

                # Shift to window boundary if within minimum block time of edge.
                # If near both boundaries, choose boundary with higher weight.
                wt_start = max_weights[time_window.start]  # weight at start
                wt_end = max_weights[time_window.end]  # weight at end
                delta_start = start - time_window.start - 1  # difference between start of window and block
                delta_end = time_window.end - end + 1  # difference between end of window and block
                if delta_start < time_window.min_slot_time and delta_end < time_window.min_slot_time:
                    if wt_start > time_window.end and wt_start > 0:
                        start = time_window.start
                        end = time_window.start + time_window.time_slots - 1
                    elif wt_end > 0:
                        start = time_window.end - time_window.time_slots + 1
                        end = time_window.end
                elif delta_start < time_window.min_slot_time and wt_start > 0:
                    start = time_window.start
                    end = time_window.start + time_window.time_slots - 1
                elif delta_end < time_window.min_slot_time and wt_start > 0:
                    start = time_window.end - time_window.time_slots + 1
                    end = time_window.end

            # If observation is already in plan, shift to side of window closest to existing obs.
            # TODO: try to shift the plan to join the pieces and save an acq
            else:
                if np.where(self.schedule[max_site] == max_idx)[0][0] < time_window.start:
                    # Existing obs in plan before window. Schedule at beginning of window.
                    start = time_window.start
                    end = time_window.start + time_window.time_slots - 1
                else:
                    # Existing obs in plan after window. Schedule at end of window.
                    start = time_window.end - time_window.time_slots + 1
                    end = time_window.end

        else:
            # If window smaller than observation length.
            start = time_window.start
            end = time_window.end

        if self.verbose:
            print('Chosen index start', start)
            print('Chosen index end', end)
            print('Current obs time: ', max_observation.observed())
            print('Current tot time: ', max_observation.length())
            input()


        # Select calibration and place/split observations
        calibrations = max_observation.calibrations
        science = max_observation.observations

        if len(calibrations) > 0: #0 need for calibration
            # How many standards needed based on science time
            std_time_slots = max_observation.standard_time  
            standards = max(1,int((time_window.length - calibrations[0].length) // std_time_slots))     

            if standards == 1:
                
                calibration, before = self._match_airmass(max_observation.site, start, end, science, 
                                                         calibrations)
                # Check for the right placement on 
                if before:
                    
                    self.plan.schedule_observation(max_site, calibration.idx, start, start+calibration.length)
                    calibration.observed+=calibration.length # Time accounting 
                    new_start = start + calibration.length

                    self.plan.schedule_observations(max_site, max_observation.observations, new_start, end)
                    
                else:
                    new_end = end - calibration.length + 1
                    self.plan.schedule_observation(max_site, calibration.idx, new_end, 
                                                   new_end+calibration.length)
                    calibration.observed+=calibration.length  # Time accounting                           
                    
                    self.plan.schedule_observations(max_site, max_observation.observations, start, new_end)
                    
            else:
             # NOTE: From Bryan's old code:
             # need two or more standards
             # if one standard, should put before and after if airmass match ok, otherwise the best one
             # the general case should handle any number of standards, splitting as needed
             # need to check that all standards are visible where placed
             # currently this just uses the first two standards defined

                first_calibration = calibrations[0]
                second_calibration = calibrations[1]
                new_start = start + first_calibration.length

                # First standard
                print(first_calibration.name, first_calibration.idx)
                self.plan.schedule_observation(max_site, first_calibration.idx, start, new_start)
                first_calibration.observed += first_calibration.length  # Time accounting  
                
                new_end = end - second_calibration.length + 1
                self.plan.schedule_observations(max_site, science, new_start, new_end)
            
                #Second standard
                self.plan.schedule_observation(max_site, second_calibration.idx, start,
                                               new_start+second_calibration.length)
                second_calibration.observed += second_calibration.length # Time accounting 
      
        else:
            # put science observations in order no need for calibrations 
            self.plan.schedule_observations(max_site, science, start, end)

        self.plan.units[max_site].append(max_observation) # add unit to plan queue
        
        
        # Number of spots in time grid used (excluding calibration).
        ntmin = np.minimum(time_window.time_slots, end - start + 1)

        # Adjust weights of scheduled observation.
        if max_observation.observed() == max_observation.length():
            # If completed, set all to negative values.
            max_weights = -1.0 * max_weights
        else:
            # If observation not fully completed, set only scheduled portion negative. Increase remaining.
            max_weights[start:end + 1] = -1.0 * max_weights[start:end + 1]
            # wpositive = np.where(max_weights > 0)[0][:]
            # max_weights[wpositive] = max_weights[wpositive] * 1.5
            # TODO: Update visfrac and weight, do outside this routine?

        # Set weights to zero for other sites so it won't be scheduled again
        for site in self.sites:
            if site != max_site:
                self.time_slots.weights[site][max_idx][:] = 0.0

        # Add new acquisition overhead to total time if observation not complete.
        max_observation.acquisition() # NOTE: this method now is specific to incomplete observations
        
        # Save changes.
        self.observations[max_observation.idx] = max_observation
        self.time_slots.weights[max_site][max_idx] = max_weights

        # TODO: This will crash as nttime, ntcal, and nobswin not defined.
        if self.verbose:
            print('Current plan: ', self.plan[max_observation.site])
            print('New obs. weights: ', max_weights)
            print('ntmin: ', ntmin)
            print('Tot time: ', max_observation.length())
            print('New obs time: ', max_observation.observed())
            print('New comp time: ', max_observation.length() - max_observation.observed())
            input()

        return True  # successfully added an observation to the plan

    def _run(self)-> NoReturn:
        """
        GreedyMax driver.
        """
        # Initialize variables.
        max_observation = None
        time_window = None

        # -- Add an observation to the plan --
        i_iter = 0
        scheduled = False

        while not scheduled:
            i_iter += 1

            if self.verbose:
                print('greedy iteration:', i_iter)

            if self.plan.timeslots_not_scheduled() != 0:

                # Find max unit to 
                max_observation, time_window = self._find_max_observation()
            
                # Place observation in schedule
                if self._insert(max_observation, time_window):
                    scheduled = True
                    print(self.time_slots.weights[max_observation.site][max_observation.idx])
                else:
                    print('No max observation picked')
            # No available spots in plan
            else:
                break

    def schedule(self):
        """
        Schedule a single night for multiple sites using the greedy-max algorithm
        """

        # ====== Initialize plan parameters ======
        # Set these initially here so that they are initialized.
        sum_score = 0.0
        time_used = 0
        n_iter = 0
        all_obs = get_observations(self.observations)

        # Unscheduled time slots.
        while self.plan.timeslots_not_scheduled() != 0:

            self._run()
            n_iter += 1

            # Fill nightly plan one observation at a time.
            sum_score = 0.0
            time_used = 0

            # Print current plan
            print(f'Iteration {n_iter:4d}')
            for site in self.sites:
                print(Site(site).name.upper())
                obs_in_plan = self.plan.get_observation_orders(site)

                output_table = []

                for observation in obs_in_plan:
                    obs_idx = observation[0]
                    start = observation[1]
                    end = observation[2]   
                    if obs_idx != EMPTY and obs_idx != UNSCHEDULABLE:
                        unit = self.plan.get_unit_by_observation(site, obs_idx) # TODO: handle None case
                        name = all_obs[obs_idx].name
                        weights = np.max(abs(self.time_slots.weights[site][unit.idx][start:end+1]))
                        category = all_obs[obs_idx].category.name
                        output_table.append([name,obs_idx,category,start, end, weights])
                        sum_score += np.sum(abs(self.time_slots.weights[site][unit.idx][start:end+1]))
                        time_used += (end - start + 1)    
                print(tabulate(output_table, headers=['Obs', 'obs_order', 'category','start', 'end', 'Max W']))
            input()

        print(f'Sum score = {sum_score:7.2f}')
        print(f'Sum score/time step = {(sum_score / (len(self.sites) * self.time_slots.total)):7.2f}') 
        print(f'Time scheduled = {(time_used * self.time_slots.slot_length.to(u.hr)):5.2f}')
       
        return self

    def print_plan(self):
        
        for site in self.sites:
            # if s == 'gs':
            #     night_length = gs_night_length[i_day]
            # else:
            #     night_length = gn_night_length[i_day]
            print(Site(site).name.upper())
            obs_order, i_start, i_end = get_order(plan=self.plan[site])
            # print(obs_order, i_start, i_end)
            sum_score = 0.0
            sum_metric = 0.0
            time_used = 0.0 * u.hr
            for i in range(len(obs_order)):
                if obs_order[i] >= 0:
                    print('{:18} {} {} {:8.4f}'.format(short_observation_id(otab_gngs['obs_id'][obs_order[i]]),
                                                    uttime_gngs[i_start[i]].strftime('%H:%M'),
                                                    uttime_gngs[i_end[i]].strftime('%H:%M'),
                                                    np.max(abs(targtab.weights[site][obs_order[i]][i_start[i]:i_end[i]]))
                                                    )
                        )
                    # print(obs_order[i], np.mean(scores[obs_order[i]][i_start[i]:i_end[i]]))
                    sum_score += np.sum(abs(targtab.weights[site][obs_order[i]][i_start[i]:i_end[i] + 1]))
                    # sum_metric += np.sum(targtab['metric'][obs_order[i]][i_start[i]:i_end[i] + 1])
                    time_used += (i_end[i] - i_start[i] + 1) * dt
            
            print('Sum score = {:7.2f}'.format(sum_score))
            print('Sum score/time step = {:7.2f}'.format(sum_score / nt))
            # print('Sum metric = {:7.2f}'.format(sum_metric))
            # print('Sum metric/time step = {:7.2f}'.format(sum_metric / nt))
            print('Time scheduled = {:5.2f}'.format(time_used))
            # print('Fraction of night scheduled = {:5.2f}'.format((time_used / night_length).value))
            print('')