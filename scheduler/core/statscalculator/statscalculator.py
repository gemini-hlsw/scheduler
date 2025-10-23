# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from collections import Counter
from datetime import timedelta
from typing import Dict, FrozenSet

import numpy as np
from lucupy.minimodel import Band, ProgramID, NightIndex, Program

from scheduler.core.components.collector import Collector
from scheduler.core.components.ranker import Ranker
from scheduler.core.events.queue import InterruptionResolutionEvent, FaultResolutionEvent, WeatherClosureResolutionEvent, \
    InterruptionEvent, FaultEvent, WeatherClosureEvent
from scheduler.core.events.queue import NightlyTimeline
from scheduler.core.plans import NightStats
from scheduler.core.statscalculator.run_summary import RunSummary, Summary
from scheduler.core.types import TimeLossType
from scheduler.graphql_mid.scalars import Sites
from scheduler.services import logger_factory


__all__ = [
    'StatCalculator',
]


logger = logger_factory.create_logger(__name__)

class StatCalculator:
    """
    Interface for stats in the calculation and results of plans.
    """

    _UNSCHEDULE_KEY: TimeLossType = 'unschedule'
    _WEATHER_KEY: TimeLossType = 'weather'
    _FAULT_KEY: TimeLossType = 'faults'


    @staticmethod
    def calculate_timeline_stats(timeline: NightlyTimeline,
                                 nights: FrozenSet[NightIndex],
                                 sites: Sites,
                                 collector: Collector,
                                 ranker: Ranker) -> RunSummary:

        # scores_per_program: Dict[ProgramID, float] = {}
        metrics_per_program: Dict[ProgramID, float] = {}
        metrics_per_band: Dict[str, float] = {}
        programs = {}

        for night_idx in nights:
            for site in sites:
                for entry in timeline.timeline[night_idx][site]:
                    # Some plans are shown empty if the telescope is closed.
                    if entry.plan_generated is None:
                        continue

                    plan = entry.plan_generated  # Update last plan

                    if 'Morning' in entry.event.description:
                        for v in plan.visits:
                            obs = collector.get_observation(v.obs_id)
                            program = collector.get_program(obs.belongs_to)

                            # Check if program is on the table
                            metrics_per_program.setdefault(program.id, 0.0)
                            metrics_per_band.setdefault(obs.band.name, 0.0)

                    # Calculate night stats for the plan
                    n_toos = 0
                    plan_score = 0
                    plan_conditions = []
                    completion_fraction: Counter[Band] = Counter({b: 0 for b in Band})

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

                        completion_fraction[obs.band] += 1

                        # Calculate altitude data
                        ti = collector.get_target_info(visit.obs_id)
                        end_time_slot = visit.start_time_slot + visit.time_slots
                        values = ti[night_idx].alt[visit.start_time_slot: end_time_slot]
                        alt_degs = [val.dms[0] + (val.dms[1] / 60) + (val.dms[2] / 3600) for val in values]
                        plan.alt_degs.append(alt_degs)

                    program_completion = {p.id: StatCalculator.calculate_program_completion(programs[p])
                                          for p in programs}
                    # TODO: this should be populated inside the plan. At runtime even.
                    plan.night_stats = NightStats(timeline.time_losses[night_idx][site],
                                                  plan_score,
                                                  n_toos,
                                                  completion_fraction,
                                                  program_completion)


        plans_summary: Summary = {}
        for p_id in metrics_per_program:
            program = collector.get_program(p_id)
            completion_bands = StatCalculator.calculate_program_completion_band(program)
            completion_str = StatCalculator.calculate_program_completion(program)
            metric_total = 0.0
            for band in completion_bands.keys():
                metric, _ = ranker.metric_slope(np.array([completion_bands[band]]),
                                            np.array([band.value]),
                                            np.array([0.8]),
                                            program.thesis)
                metrics_per_band[band.name] += metric[0]
                # For now sum all the metrics
                metric_total += metric[0]
            plans_summary[p_id.id] = (completion_str, metric_total)

        return RunSummary(plans_summary, metrics_per_band)

    @staticmethod
    def calculate_program_completion(program: Program) -> str:

        total_used = program.program_used()
        prog_total = program.program_awarded()
        return f'{float((total_used.total_seconds()) / prog_total.total_seconds()) * 100:.1f}%'

    @staticmethod
    def calculate_program_completion_band(program: Program) -> str:
        """Completion by band"""

        completion = {}
        for band in program.bands():
            total_used = program.program_used(band=band)
            prog_total = program.program_awarded(band=band)
            completion[band] = total_used.total_seconds() / prog_total.total_seconds()

        return completion