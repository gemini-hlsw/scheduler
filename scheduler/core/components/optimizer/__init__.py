# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import List

from scheduler.core.calculations.selection import Selection
from scheduler.core.plans import Plans

import numpy.typing as npt
from lucupy.minimodel import Program

# Convenient type alias for Interval
Interval = npt.NDArray[int]


class Optimizer:
    """
    Entrypoint to interact with an BaseOptimizer object.
    All algorithms need to follow the same structure to create a Plan
    """

    def __init__(self, algorithm=None):
        self.algorithm = algorithm
        self.selection = None
        self.period = None
        self.night_events = None

    def schedule(self, selection: Selection) -> List[Plans]:
        # Create set of plans for the amount of nights
        self.selection = selection
        self.algorithm.setup(selection)
        self.night_events = selection.night_events
        # TODO: Assumes that all sites schedule the same amount of nights
        # if num_nights_optimize is None:
        # self.period = len(list(self.night_events.values())[0].time_grid)
        nights = [Plans(self.night_events, night) for night in range(self.period)]
        self.algorithm.schedule(nights)
        return nights

    def _update_score(self, program: Program) -> None:
        """Update the scores of the incomplete groups in the scheduled program"""
        program_calculations = self.selection.score_program(program)

        for unique_group_id in program_calculations.top_level_groups:
            group_data = program_calculations.group_data_map[unique_group_id]
            group, group_info = group_data
            schedulable_group = self.selection.schedulable_groups[unique_group_id]
            # update scores in schedulable_groups if the group is not completely observed
            if schedulable_group.group.exec_time() >= schedulable_group.group.total_used():
                schedulable_group.group_info.scores[:] = group_info.scores[:]