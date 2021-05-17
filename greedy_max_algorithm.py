from greedy_max import *
from schedule import *
import numpy as np
import astropy.units as u
from astropy.units.quantity import Quantity
from typing import List, Optional, Tuple


class GreedyMax:
    def __init__(self, obs: List[Observation], time_slots: TimeSlots, sites: List[Site],
                 tmin: Quantity = 30.0 * u.min, verbose: bool = False):
        self.plan = Plan(time_slots.total, sites)
        self.observations = obs
        self.time_slots = time_slots
        self.tmin = tmin
        self.sites = sites
        self.verbose = verbose
        self.min_slot_time = int(np.ceil(tmin.to(u.h) / time_slots.slot_length.to(u.h)))

    # TODO: What does this do?
    def _reset_slot_time(self):
        self.min_slot_time = int(np.ceil(self.tmin.to(u.h) / self.time_slots.slot_length.to(u.h)))

    # TODO: Move this method to scheduling groups
    # TODO: This method may be static.
    # TODO: This method is not in use.
    # @staticmethod
    # def _time_calibration(self, inst: str, disperser: str) -> Quantity:
    #     """
    #     Return the time needed for calibrations (esp. telluric standards)
    #     """
    #     dl = disperser.lower()
    #     return (18.00 if 'GMOS' not in inst and 'mirror' not in dl and 'null' not in dl else 0.0) * u.min

    # TODO: This method may be static.
    @staticmethod
    def _acquisition_overhead_time(self, disperser: str) -> Quantity:
        """
        Get acquisition overhead time
        """
        if 'mirror' in disperser.lower():
            return 0.2 * u.h
        else:
            return 0.3 * u.h

    def _find_max_observation(self) -> Tuple[
                                            Optional[Observation],
                                            Dict[Site, np.ndarray],
                                            np.ndarray
                                        ]:
        """
        Select the observation with the maximum weight in a time interval

        Returns
        -------
        max_obs : Observation object (or None)

        intervals : time intervals array for that observation

        iwinmax : time window indices
        """
        max_weight = 0.  # maximum weight in time interval
        intervals = {}  # save intervals for each site for later use

        iwinmax = None
        max_obs = None

        for site in self.sites:
            empty_slots = self.plan.empty_slots(site)

            if len(empty_slots) != 0:
                indx = self.time_slots.intervals(empty_slots)  # intervals of empty time slots
                iint = empty_slots[np.where(indx == 1)[0][:]]  # first interval of indx
                intervals[site] = iint

                if self.verbose:
                    print('Empty slots:', empty_slots)
                    print('indx', indx)
                    print('iint: ', iint)
                    print('site: ', site)

                for observation in self.observations:
                    # Get the maximum weight in the interval.
                    wmax = np.max(self.time_slots.weights[site][observation.idx][iint])

                    # Get the indices of weights > 0 in the interval.
                    ipos = np.where(self.time_slots.weights[site][observation.idx][iint] > 0)[0][:]

                    # Test
                    # TODO: What is this testing? We should document this.
                    if len(ipos) > 0 and wmax > max_weight:
                        iwinmax = iint[ipos]  # indices with pos. weights within first empty window
                        max_weight = wmax

                        observation.site = site
                        max_obs = observation

                        if self.verbose:
                            print('maxweight', max_weight)
                            print('max obs: ', observation.name)
                            print('iimax', observation.idx)
                            print('smax', observation.site)
                            input()

        return max_obs, intervals, iwinmax

    def _insert(self, max_observation: Optional[Observation], time_window: Optional[TimeWindow]) -> bool:
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

        if not max_observation or not TimeWindow:
            for site in self.sites:
                if len(time_window.intervals[site]) > 0:
                    self.plan.schedule[site][time_window.intervals[site]] = UNSCHEDULABLE
            return False

        # Place observation within available window --
        max_site = max_observation.site
        max_idx = max_observation.idx
        max_weights = self.time_slots.weights[max_site][max_idx]

        if 0 < time_window.total_time <= time_window.nobswin:
            if not self.plan.is_observation_scheduled(max_site, max_idx):
                # Determine schedule placement for maximum integrated weight
                maxf = 0.0
                # TODO: I'm not clear on this documentation.
                if time_window.total_time > 1:
                    # NOTE: integrates over one extra time slot...
                    # ie. if nttime = 14, then the program will choose 15
                    # x values to do trapz integration (therefore integrating
                    # 14 time slots).
                    if self.verbose:
                        print('\nIntegrating max obs. over window...')
                        print('wstart', time_window.start)
                        print('wend', time_window.end)
                        print('nttime', time_window.total_time)
                        print('j values', np.arange(time_window.start, time_window.end - time_window.total_time + 2))

                    for j in range(time_window.start, time_window.end - time_window.total_time + 2):
                        f = sum(max_weights[j:j + time_window.total_time])

                        # TODO: Error here: nttime is not defined. Is it time_window.total_time?
                        # TODO: This will crash if verbose is on.
                        # TODO: The only place I see nttime defined is in the _run method as a local variable.
                        if self.verbose:
                            print('j range', j, j + nttime - 1)
                            print('obs weight', max_weights[j:j + nttime])
                            print('integral', f)

                        if f > maxf:
                            maxf = f
                            start = j
                            end = start + time_window.total_time - 1
                else:
                    # TODO: Error here: iwinmax is not defined. Again, only defined in _run method as local variable.
                    # TODO: This will crash if it is reached.
                    start = np.argmax(max_weights[iwinmax])
                    maxf = np.amax(max_weights[start])
                    end = start + time_window.total_time - 1

                # TODO: If else code is reached above, start and end will not be defined, and the code will crash.
                if self.verbose:
                    print('max integral of weight func (maxf)', maxf)
                    print('index start', start)
                    print('index end', end)

                # Shift to start or end of night if within minimum block time from boundary.
                # Nudge:
                if start < self.min_slot_time:
                    if self.plan[max_site][0] == -1 and max_weights[0] > 0:
                        start = 0
                        end = start + time_window.total_time - 1
                elif self.time_slots.total - end < self.min_slot_time:
                    if self.plan[max_site][-1] == -1 and max_weights[-1] > 0:
                        end = self.time_slots.total - 1
                        start = end - time_window.total_time + 1

                # Shift to window boundary if within minimum block time of edge.
                # If near both boundaries, choose boundary with higher weight.
                wt_start = max_weights[time_window.start]  # weight at start
                wt_end = max_weights[time_window.end]  # weight at end
                delta_start = start - time_window.start - 1  # difference between start of window and block
                delta_end = time_window.end - end + 1  # difference between end of window and block
                if delta_start < self.min_slot_time and delta_end < self.min_slot_time:
                    if wt_start > time_window.end and wt_start > 0:
                        start = time_window.start
                        end = time_window.start + time_window.total_time - 1
                    elif wt_end > 0:
                        start = time_window.end - time_window.total_time + 1
                        end = time_window.end
                elif delta_start < self.min_slot_time and wt_start > 0:
                    start = time_window.start
                    end = time_window.start + time_window.total_time - 1
                elif delta_end < self.min_slot_time and wt_start > 0:
                    start = time_window.end - time_window.total_time + 1
                    end = time_window.end

            # If observation is already in plan, shift to side of window closest to existing obs.
            # TODO: try to shift the plan to join the pieces and save an acq
            else:
                if np.where(self.schedule[max_site] == max_idx)[0][0] < time_window.start:
                    # Existing obs in plan before window. Schedule at beginning of window.
                    start = time_window.start
                    end = time_window.start + time_window.total_time - 1
                else:
                    # Existing obs in plan after window. Schedule at end of window.
                    start = time_window.end - time_window.total_time + 1
                    end = time_window.end

        else:
            # If window smaller than observation length.
            start = time_window.start
            end = time_window.end

        if self.verbose:
            print('Chosen index start', start)
            print('Chosen index end', end)
            print('Current obs time: ', max_observation.time)
            print('Current tot time: ', max_observation.total_time)
            input()

        self.plan.schedule_observation(max_site, max_idx, start, end + 1)

        # Number of spots in time grid used (excluding calibration).
        ntmin = np.minimum(time_window.total_time - time_window.calibration_time,
                           end - start + 1)

        # Update time.
        max_observation.time += self.time_slots.slot_length.to(u.h) * ntmin

        # Update completion fraction.
        max_observation.completion += self.time_slots.slot_length.to(u.h) * ntmin / max_observation.total_time

        # Adjust weights of scheduled observation.
        if max_observation.completion >= 1:
            # If completed, set all to negative values.
            max_weights = -1.0 * max_weights
        else:
            # If observation not fully completed, set only scheduled portion negative. Increase remaining.
            max_weights[start:end + 1] = -1.0 * max_weights[start:end + 1]
            wpositive = np.where(max_weights > 0)[0][:]
            max_weights[wpositive] = max_weights[wpositive] * 1.5
            # TODO: Update visfrac and weight, do outside this routine?

        # Set weights to zero for other sites so it won't be scheduled again
        for site in self.sites:
            if site != max_site:
                self.time_slots.weights[site][max_idx][:] = 0.0

        # Add to total time if observation not complete.
        if max_observation.disperser and max_observation.time < max_observation.total_time:
            acqover = GreedyMax._acquisition_overhead_time(max_observation.disperser)
            max_observation.total_time += acqover

        # Save changes.
        self.observations[max_observation.idx] = max_observation
        self.time_slots.weights[max_site][max_idx] = max_weights

        # TODO: This will crash as nttime, ntcal, and nobswin not defined.
        if self.verbose:
            print('Current plan: ', self.plan[max_observation.site])
            print('New obs. weights: ', max_weights)
            print('nttime - ntcal , nobswin: ', nttime - ntcal, nobswin)
            print('ntmin: ', ntmin)
            print('Tot time: ', max_observation.total_time)
            print('New obs time: ', max_observation.time)
            print('New comp time: ', max_observation.completion)
            input()

        return True  # successfully added an observation to the plan

    def _run(self):
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
                # Try to schedule an observation.
                i_gow = 0
                can_be_scheduled = False
                while not can_be_scheduled:
                    i_gow += 1

                    if self.verbose:
                        print('i_gow', i_gow)

                    self._reset_slot_time()
                    max_observation, intervals, iwinmax = self._find_max_observation()

                    # Determine observation window and length if there is an observation.
                    if not max_observation:
                        break

                    # Boundaries of available window
                    wstart = iwinmax[0]  # window start
                    wend = iwinmax[-1]   # window end
                    nobswin = wend - wstart + 1

                    # Calibration time
                    ntcal = max_observation.calibrate(self.time_slots.slot_length)

                    # Remaining time (including calibration)
                    ttime = (max_observation.total_time - max_observation.time)

                    # Number of slots needed in time grid, rounding up.
                    nttime = int(np.ceil((ttime.to(u.h) / self.time_slots.slot_length.to(u.h)))) + ntcal

                    # Don't leave little pieces of observations remaining.
                    # Also, short observations are done entirely.
                    if nttime - self.min_slot_time <= self.min_slot_time:
                        self.min_slot_time = nttime

                    time_window = TimeWindow(wstart, wend, nobswin, nttime, ntcal, intervals)

                    if self.verbose:
                        print('ID of chosen ob.', max_observation.name)
                        print('weights of chosen ob.',
                              self.time_slots.weights[max_observation.site][max_observation.idx])
                        print('Current plan', self.plan[max_observation.site])
                        print('wstart', wstart)
                        print('wend', wend)
                        print('dt', self.time_slots.slot_length)
                        print('tot_time', max_observation.total_time)
                        print('obs_time', max_observation.time)
                        print('ttime', ttime)
                        print('nttime', nttime)
                        print('nobswin', nobswin)
                        print('nminuse', self.min_slot_time)
                        input()

                    # Decide whether or not to add to schedule.
                    if np.logical_or(nttime <= nobswin, nobswin >= self.min_slot_time):
                        # Schedule the observation.
                        can_be_scheduled = True
                    else:  # Do not schedule observation
                        self.time_slots.weights[max_observation.site][max_observation.idx][
                            intervals[max_observation.site]] = 0
                        if self.verbose:
                            print('Block too short to schedule...')

                # TODO: If while loop above is not executed, max_observation and time_window will be unassigned?
                # TODO: Will this crash?
                # Place observation in schedule
                if self._insert(max_observation, time_window):
                    scheduled = True
            # No available spots in plan
            else:
                break

        return self.plan, self.observations, self.time_slots

    def schedule(self):
        """
        Schedule a single night for multiple sites using the greedy-max algorithm
        """

        # ====== Initialize plan parameters ======
        # Set these initially here so that they are initialized.
        sum_score = 0.0
        time_used = 0
        n_iter = 0

        # Unscheduled time slots.
        while self.plan.timeslots_not_scheduled() != 0:

            plan, obstab, targtab = self._run()
            n_iter += 1

            # Fill nightly plan one observation at a time.
            sum_score = 0.0
            time_used = 0

            # Print current plan
            # TODO: Move this to inside of plan. as a summary method
            print('Iteration {:4d}'.format(n_iter))
            for site in self.sites:
                print(Site(site).name.upper())
                print('{:18} {:>9} {:>8} {:>8} {:>8}'.format('Obsid', 'obs_order', 'i_start', 'i_end', 'Max W'))
                obs_order, i_start, i_end = get_order(plan=plan.schedule[site])

                for i in range(len(obs_order)):
                    if obs_order[i] >= 0:
                        print('{:18} {:>9d} {:>8d} {:>8d} {:8.4f}'.format(short_observation_id(self.observations[obs_order[i]].name),
                                                                          obs_order[i], i_start[i], i_end[i],
                                                                          np.max(abs(
                                                                              self.time_slots.weights[site][
                                                                                  obs_order[i]][
                                                                                  i_start[i]:i_end[i] + 1]))))
                        sum_score += np.sum(abs(self.time_slots.weights[site][obs_order[i]][i_start[i]:i_end[i] + 1]))
                        time_used += (i_end[i] - i_start[i] + 1)
            # input()

        print('Sum score = {:7.2f}'.format(sum_score))
        print('Sum score/time step = {:7.2f}'.format(sum_score / (2 * self.time_slots.total)))
        print('Time scheduled = {:5.2f}'.format(time_used * self.time_slots.slot_length.to(u.hr)))

        return self
