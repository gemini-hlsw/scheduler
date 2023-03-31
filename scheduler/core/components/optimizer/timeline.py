# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import ClassVar, List, NoReturn, Mapping, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from lucupy.minimodel import Observation, ObservationID, Site

from scheduler.core.calculations.nightevents import NightEvents

# Type alias for Interval for simplicity.
Interval = npt.NDArray[int]


@dataclass
class Timeline:
    """
    Nightly plan for a specific Site for the GreedyMax optimizer. Each plan is a time_slots array with one
    entry for each time slot for the night. Each value needs to be an index to the observation
    scheduled in that slot.
    """
    start: datetime
    end: datetime
    time_slot_length: timedelta
    site: Site
    total_time_slots: int

    EMPTY: ClassVar[int] = -1
    UNSCHEDULABLE: ClassVar[int] = -2

    def __post_init__(self):
        self.time_slots = np.full(self.total_time_slots, Timeline.EMPTY)
        self.is_full = False

    def __contains__(self, obs: Observation) -> bool:
        return obs.id in self.time_slots

    def _empty_slots(self) -> npt.NDArray[int]:
        """
        Determine the number of empty time slots.
        """
        return np.where(self.time_slots == Timeline.EMPTY)[0][:]

    @staticmethod
    def _available_intervals(empty_slots: np.ndarray) -> npt.NDArray[Interval]:
        """
        Calculate the available intervals in the schedule by creating an array that contains all
        the groups of consecutive numbers
        """
        return np.split(empty_slots, np.where(np.diff(empty_slots) != 1)[0] + 1)

    def get_available_intervals(self, first: bool = False) -> Union[Interval, npt.NDArray[Interval]]:
        """
        Get the set of time_slot intervals that can be scheduled. If desired, return only the first
        """
        empty_slots = self._empty_slots()
        intervals = self._available_intervals(empty_slots)
        # print(f"length intervals = {len(intervals)}")
        return intervals[0] if first and len(intervals) > 1 else intervals

    def get_earliest_available_interval(self) -> Interval:
        """
        Get the earliest available space in the schedule that can allocate an observation
        """
        return self.get_available_intervals()[0]

    def add(self, iobs: int, time_slots: int, interval: Interval) -> datetime:
        """
        Add an observation index to the first open position (-1) in the given interval
        Returns the time of this position
        """
        # interval = self.get_earliest_available_interval()
        # Get first non-zero slot in given interval
        i_first = np.where(self.time_slots[interval] == Timeline.EMPTY)[0][0]

        # Set values of time_slots to the observation index
        self.time_slots[interval[i_first:i_first + time_slots]] = iobs

        # Clock time for the starting index
        start = self.start + interval[i_first] * self.time_slot_length

        return start

    def get_observation_order(self) -> List[Tuple[int, int, int]]:
        """
        Get the observation idx and position for all the schedule observations in order
        Return
        -------

        orders: List of a 3-dimensional tuple: observation.idx, initial position
                and last position in the plan.
        """

        obs_comparator = self.time_slots[0]
        start = 0
        order = []
        for position, obs_idx in enumerate(self.time_slots):
            if obs_idx != obs_comparator:
                order.append((obs_comparator, start, position - 1))
                start = position
                obs_comparator = obs_idx
            elif position == len(self.time_slots) - 1:
                order.append((obs_comparator, start, position))

        return order

    def __str__(self):
        """Print the result of get_observation_order, an experiment in the use of __str__"""

        obs_order = self.get_observation_order()

        return f"{obs_order}"

    def print(self, obs_ids: Sequence[ObservationID]) -> NoReturn:
        """Print the obsids and times associated with the timeline"""

        delta = timedelta(milliseconds=500)  # for rounding to the nearest second

        obs_order = self.get_observation_order()

        for idx, i_start, i_end in obs_order:
            obs_id = obs_ids[idx]
            if idx > -1:
                # Convert time slot indices to UT
                t_start = self.start.astimezone(tz=timezone.utc) + i_start * self.time_slot_length + delta
                # t_end = self.start.astimezone(tz=timezone.utc) + i_end * self.time_slot_length + delta
                t_end = t_start + (i_end - i_start) * self.time_slot_length

                print(f'{idx:5d} {i_start:5d} {i_end:5d}   {obs_id:20} {t_start.strftime("%Y-%m-%d %H:%M:%S")} '
                      f'{t_end.strftime("%Y-%m-%d %H:%M:%S")}')


class Timelines:
    """
    A collection of Timeline from all sites for a specific night
    """

    def __init__(self, night_events: Mapping[Site, NightEvents], night_idx: int):

        self.timelines = {}
        self.night = night_idx
        for site, ne in night_events.items():
            if ne is not None:
                self.timelines[site] = Timeline(ne.local_times[night_idx][0],
                                                ne.local_times[night_idx][-1],
                                                ne.time_slot_length.to_datetime(),
                                                site,
                                                len(ne.times[night_idx]))

    def __getitem__(self, site: Site) -> Timeline:
        return self.timelines[site]

    def __iter__(self):
        return iter(self.timelines.values())

    def all_done(self) -> bool:
        """
        Check if all plans for all sites are done in a night
        """
        return all(timeline.is_full for timeline in self.timelines.values())