# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import time
from dataclasses import dataclass
from typing import final

import numpy.typing as npt
from lucupy.minimodel import Site, NightIndex, TimeslotIndex

from scheduler.core.components.collector import Collector
from scheduler.core.components.optimizer import Optimizer
from scheduler.core.components.ranker import Ranker
from scheduler.core.components.selector import Selector
from scheduler.core.plans import Plans
from scheduler.services import logger_factory

_logger = logger_factory.create_logger(__name__)


@final
@dataclass
class SCP:
    """
    Scheduler Core Pipeline.
    This process must remain the same across all modes and methods of consume
    the scheduler.
    """
    collector: Collector
    selector: Selector
    optimizer: Optimizer

    def run(self,
            site: Site,
            night_indices: npt.NDArray[NightIndex],
            current_timeslot: TimeslotIndex,
            ranker: Ranker) -> Plans:
        ts0 = time.time()
        selection = self.selector.select(night_indices=night_indices,
                                         sites=frozenset([site]),
                                         starting_time_slots={site: {night_idx: current_timeslot
                                                                     for night_idx in night_indices}},
                                         ranker=ranker)
        print(f'\t\tSelection created in {time.time() - ts0} sec')

        # Right now the optimizer generates List[Plans], a list of plans indexed by
        # every night in the selection. We only want the first one, which corresponds
        # to the current night index we are looping over.
        # _logger.debug(f'Running optimizer for {site.site_name} for night {night_idx} '
        #               f'starting at time slot {current_timeslot}.')
        top0 = time.time()
        plans = self.optimizer.schedule(selection)[0]
        print(f'\t\tGM plan created in {time.time() - top0} sec')
        return plans
