# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os

from typing import FrozenSet
from astropy.time import Time
from lucupy.minimodel import Site, Semester

from scheduler.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from scheduler.core.builder import SchedulerBuilder, Blueprints
from scheduler.core.statscalculator import StatCalculator
from scheduler.db.planmanager import PlanManager

from definitions import ROOT_DIR


class Service:
    def __init__(self,
                 start_time: Time,
                 end_time: Time,
                 num_nights_to_schedule: int,
                 semesters: FrozenSet[Semester],
                 sites: FrozenSet[Site],
                 builder: SchedulerBuilder):
        self.start_time = start_time
        self.end_time = end_time
        self.num_nights_to_schedule = num_nights_to_schedule
        self.semesters = semesters
        self.sites = sites
        self.builder = builder

    def __call__(self):
        # signal.signal(signal.SIGINT, signal.SIG_IGN)
        # builder = SchedulerBuilder()  # To trigger the decorator
        programs = read_ocs_zipfile(os.path.join(ROOT_DIR, 'scheduler', 'data', '2018B_program_samples.zip'))

        # Retrieve observations from Collector
        collector = self.builder.build_collector(self.start_time,
                                                 self.end_time,
                                                 self.sites,
                                                 self.semesters,
                                                 Blueprints.collector)
        collector.load_programs(program_provider_class=OcsProgramProvider,
                                data=programs)
        # Create selection from Selector
        selector = self.builder.build_selector(collector, self.num_nights_to_schedule)
        selection = selector.select(sites=self.sites)

        # Execute the Optimizer.
        optimizer = self.builder.build_optimizer(Blueprints.optimizer)
        plans = optimizer.schedule(selection)

        # Calculate plans stats
        plan_summary = StatCalculator.calculate_plans_stats(plans, collector)

        # Save to database
        PlanManager.set_plans(plans, self.sites)
        return plans, plan_summary


def build_service(start: Time,
                  end: Time,
                  num_nights_to_schedule: int,
                  sites: FrozenSet[Site],
                  builder: SchedulerBuilder) -> Service:
    """

    Args:
        start (Time): Astropy start time for calculations.
        end (Time): Astropy end time for calculations.
        num_nights_to_schedule (int): The number of nights for which to generate plans starting at start.
        sites: (FrozenSet[Site]) = Sites to do the schedule.
        builder: (SchedulerBuilder) = Builder to create Scheduler components.

    Returns:
        Scheduler: Callable executed in the ProcessManager
    """
    semesters = frozenset([Semester.find_semester_from_date(start.to_value('datetime')),
                           Semester.find_semester_from_date(end.to_value('datetime'))])

    return Service(start, end, num_nights_to_schedule, semesters, sites, builder)
