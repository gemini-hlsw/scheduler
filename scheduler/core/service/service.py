# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
import signal

from typing import FrozenSet
from astropy.time import Time
from lucupy.minimodel import Site, ALL_SITES

from scheduler.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from scheduler.core.builder import SchedulerBuilder, Blueprints
from scheduler.db.planmanager import PlanManager
from definitions import ROOT_DIR


class Service:
    def __init__(self, start_time: Time, end_time: Time, sites: FrozenSet[Site]):
        self.start_time = start_time
        self.end_time = end_time
        self.sites = sites
 
    def __call__(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        builder = SchedulerBuilder()  # To trigger the decorator
        programs = read_ocs_zipfile(os.path.join(ROOT_DIR, 'scheduler', 'data', '2018B_program_samples.zip'))
        
        # Retrieve observations from Collector
        collector = builder.build_collector(self.start_time, self.end_time, self.sites, Blueprints.collector)
        collector.load_programs(program_provider_class=OcsProgramProvider,
                                data=programs)
        # Create selection from Selector
        selector = builder.build_selector(collector)
        selection = selector.select()

        # Execute the Optimizer.
        optimizer = builder.build_optimizer(selection, Blueprints.optimizer)
        plans = optimizer.schedule()

        # Save to database
        PlanManager.set_plans(plans)


def build_scheduler(start: Time = Time("2018-10-01 08:00:00", format='iso', scale='utc'),
                    end: Time = Time("2018-10-03 08:00:00", format='iso', scale='utc'),
                    sites: FrozenSet[Site] = ALL_SITES) -> Service:
    """

    Args:
        start (Time, optional): _description_. Defaults to Time("2018-10-01 08:00:00", format='iso', scale='utc').
        end (Time, optional): _description_. Defaults to Time("2018-10-03 08:00:00", format='iso', scale='utc').
        sites: (FrozenSet[Site], optional)=. Defaults to ALL_SITE

    Returns:
        Scheduler: Callable executed in the ProcessManager
    """
    return Service(start, end, sites)
