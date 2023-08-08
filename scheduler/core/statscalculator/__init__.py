from typing import Dict, List
from scheduler.core.plans import NightStats, Plans
from lucupy.minimodel import Band, Conditions, ProgramID
from scheduler.core.components.collector import Collector


class StatCalculator:
    """
    Interface for stats in the calculation and results of plans.
    """
    @staticmethod
    def calculate_plans_stats(all_plans: List[Plans],
                              collector: Collector) -> dict[str, tuple[str, float]]:

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
                    alt_degs = [val.dms[0] + (val.dms[1] / 60) + (val.dms[2] / 3600) for val in values]
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
            completion = f'{all_programs_visits[p_id] / all_programs_length[p_id] * 100:.1f}%'
            score = all_programs_scores[p_id]
            plans_summary[p_id.id] = (completion, score)

        return plans_summary
