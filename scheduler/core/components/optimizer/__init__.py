# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import List

from scheduler.core.calculations.selection import Selection
from scheduler.core.plans import Plans

import numpy.typing as npt

# Convenient type alias for Interval
Interval = npt.NDArray[int]


class Optimizer:
    """
    Entrypoint to interact with an BaseOptimizer object.
    All algorithms need to follow the same structure to create a Plan
    """

    def __init__(self, selection: Selection, algorithm=None):
        self.algorithm = algorithm.setup(selection)
        self.night_events = selection.night_events
        # TODO: Assumes that all sites schedule the same amount of nights
        # if num_nights_optimize is None:
        self.period = len(list(self.night_events.values())[0].time_grid)
        # else:
        #     self.period = num_nights_optimize

    def schedule(self) -> List[Plans]:
        # Create set of plans for the amount of nights
        nights = [Plans(self.night_events, night) for night in range(self.period)]
        self.algorithm.schedule(nights)
        return nights
