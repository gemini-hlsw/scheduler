# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os

from typing import Dict, FrozenSet, Mapping, List, Tuple
from astropy.time import Time
from lucupy.minimodel import Site, Semester, Band, Conditions, ProgramID


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
        plan_summary = calculate_plans_stats(plans, collector, selection)

        # Save to database
        PlanManager.set_plans(plans, self.sites)
        return plans, plan_summary


def build_scheduler(start: Time,
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


def calculate_plans_stats(all_plans: List[Plans],
                          collector: Collector,
                          selection: Selection) -> Mapping[ProgramID, Tuple[str, float]]:

    all_programs_visits: Dict[ProgramID, int] = {}
    all_programs_length: Dict[ProgramID, int] = {}
    all_programs_scores: Dict[ProgramID, float] = {}
    n_toos = 0
    plan_conditions = []
    completion_fraction = {b: 0 for b in Band}
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
                # check completion
                program = collector.get_program(obs.belongs_to)

                if program.id in all_programs_visits:
                    all_programs_visits[program.id] += 1
                    all_programs_scores[program.id] += visit.score
                else:
                    # TODO: This assumes the observations are not splittable
                    # TODO: and will change in the future.
                    all_programs_length[program.id] = len(program.observations())
                    all_programs_visits[program.id] = 0
                    all_programs_scores[program.id] = visit.score

                if program.band in completion_fraction:
                    completion_fraction[program.band] += 1
                else:
                    raise KeyError(f'Missing band {program.band} in program {program.id.id}.')

                # Calculate altitude data
                ti = collector.get_target_info(visit.obs_id)
                end_time_slot = visit.start_time_slot + visit.time_slots
                values = ti[plans.night_idx].alt[visit.start_time_slot: end_time_slot]
                alt_degs = [val.dms[0] + (val.dms[1]/60) + (val.dms[2]/3600) for val in values]
                plan.alt_degs.append(alt_degs)

            plan.night_stats = NightStats(f'{plan.time_left()} min',
                                            plan_score,
                                            Conditions.most_restrictive_conditions(plan_conditions),
                                            n_toos,
                                            completion_fraction)
            n_toos = 0
            plan_score = 0
            plan_conditions = []
            completion_fraction = {b: 0 for b in Band}

    plans_summary = {}
    for p_id in all_programs_visits:
        completion = f'{all_programs_visits[p_id]/all_programs_length[p_id]*100:.1f}%'
        score = all_programs_scores[p_id]
        plans_summary[p_id] = (completion, score)

    return plans_summary
