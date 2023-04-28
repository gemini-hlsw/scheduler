# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
import signal

from typing import FrozenSet, Mapping, List, Tuple
from astropy.time import Time
from lucupy.minimodel import Site, ALL_SITES, Semester, Band, Conditions, ProgramID


from scheduler.core.programprovider.ocs import read_ocs_zipfile, OcsProgramProvider
from scheduler.core.builder import SchedulerBuilder, Blueprints
from scheduler.core.components.collector import Collector
from scheduler.core.calculations.selection import Selection
from scheduler.core.plans import NightStats, Plans
from scheduler.db.planmanager import PlanManager

from definitions import ROOT_DIR


class Service:
    def __init__(self,
                 start_time: Time,
                 end_time: Time,
                 semesters: FrozenSet[Semester],
                 sites: FrozenSet[Site]):
        self.start_time = start_time
        self.end_time = end_time
        self.semesters = semesters
        self.sites = sites


    def __call__(self):
        # signal.signal(signal.SIGINT, signal.SIG_IGN)
        builder = SchedulerBuilder()  # To trigger the decorator
        programs = read_ocs_zipfile(os.path.join(ROOT_DIR, 'scheduler', 'data', '2018B_program_samples.zip'))

        # Retrieve observations from Collector
        collector = builder.build_collector(self.start_time,
                                            self.end_time,
                                            self.sites,
                                            self.semesters,
                                            Blueprints.collector)
        collector.load_programs(program_provider_class=OcsProgramProvider,
                                data=programs)
        # Create selection from Selector
        selector = builder.build_selector(collector)
        selection = selector.select(sites=self.sites)

        # Execute the Optimizer.
        optimizer = builder.build_optimizer(selection, Blueprints.optimizer)
        plans = optimizer.schedule()

        #Calculate plans stats
        plan_summary = calculate_plans_stats(plans, collector, selection)

        # Save to database
        PlanManager.set_plans(plans, self.sites)
        return plans, plan_summary


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
    semesters = frozenset([Semester.find_semester_from_date(start.to_value('datetime')),
                           Semester.find_semester_from_date(end.to_value('datetime'))])
    return Service(start, end, semesters, sites)


def calculate_plans_stats(all_plans: List[Plans],
                          collector: Collector,
                          selection: Selection) -> Mapping[ProgramID,Tuple[str,float]]:

    all_programs_visits = {}
    all_programs_length = {}
    all_programs_scores = {}
    n_toos = 0
    plan_conditions = []
    completion_fraction = {b:0 for b in Band}
    plan_score = 0

    for plans in all_plans:
        for plan in plans:
            for visit in plan.visits:
                obs = collector.get_observation(visit.obs_id)
                # check if obs is a too
                if obs.too_type is not None:
                    n_toos += 1
                # add to plan_score
                plan_score += visit.score
                # add used conditions
                plan_conditions.append(obs.constraints.conditions)
                # check completition
                program = collector.get_program(obs.belongs_to)

                if program.id in all_programs_visits:
                    all_programs_visits[program.id.id] += 1
                    all_programs_scores[program.id.id] += visit.score
                else:
                    # TODO: This asssumes the observations are not splittable
                    # TODO: and would change in the future.
                    all_programs_length[program.id.id] = len(program.observations())
                    all_programs_visits[program.id.id] = 0
                    all_programs_scores[program.id.id] = visit.score

                if program.band in completion_fraction:
                    completion_fraction[program.band] += 1
                else:
                    raise KeyError('Missing Band in Program!')
            plan.night_stats =  NightStats(f'{plan.time_left()} min',
                                            plan_score,
                                            Conditions.most_restrictive_conditions(plan_conditions),
                                            n_toos,
                                            completion_fraction)
            n_toos = 0
            plan_score = 0
            plan_conditions = []
            completion_fraction = {b:0 for b in Band}

    plans_summary = {}
    for p_id in all_programs_visits:
        completition = f'{all_programs_visits[p_id]/all_programs_length[p_id]*100:.1f}%'
        score = all_programs_scores[p_id]
        plans_summary[p_id] = (completition,score)

    return plans_summary




