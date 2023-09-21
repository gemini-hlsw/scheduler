# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from collections import Counter
from typing import Dict, List, Tuple

from lucupy.minimodel import Band, Conditions, ProgramID
from lucupy.types import ZeroTime

from scheduler.core.components.collector import Collector
from scheduler.core.plans import NightStats, Plans
from scheduler.services import logger_factory

logger = logger_factory.create_logger(__name__)


class StatCalculator:
    """
    Interface for stats in the calculation and results of plans.
    """
    @staticmethod
    def calculate_plans_stats(all_plans: List[Plans],
                              collector: Collector) -> Dict[str, Tuple[str, float]]:

        all_programs_scores: Dict[ProgramID, float] = {}

        for plans in all_plans:
            for plan in plans:
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

                    all_programs_scores.setdefault(program.id, 0)
                    all_programs_scores[program.id] += visit.score
                    completion_fraction[program.band] += 1

                    # Calculate altitude data
                    ti = collector.get_target_info(visit.obs_id)
                    end_time_slot = visit.start_time_slot + visit.time_slots
                    values = ti[plans.night_idx].alt[visit.start_time_slot: end_time_slot]
                    alt_degs = [val.dms[0] + (val.dms[1] / 60) + (val.dms[2] / 3600) for val in values]
                    plan.alt_degs.append(alt_degs)

                plan.night_stats = NightStats(f'{plan.time_left()} min',
                                              plan_score,
                                              Conditions.most_restrictive_conditions(plan_conditions),
                                              n_toos,
                                              completion_fraction)

        plans_summary = {}
        for p_id in all_programs_scores:
            program = collector.get_program(p_id)
            total_used = program.total_used()
            prog_total = sum((o.part_time() + o.acq_overhead + o.prog_time() for o in program.observations()),
                             start=ZeroTime)

            completion = f'{float(total_used.total_seconds()/prog_total.total_seconds())* 100:.1f}%'
            score = all_programs_scores[p_id]
            print(completion, score)
            plans_summary[p_id.id] = (completion, score)

        return plans_summary
