# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from collections import Counter
from typing import Dict, Tuple, FrozenSet, TypeAlias

from lucupy.minimodel import Band, ProgramID, NightIndex, Program
from lucupy.types import ZeroTime

from scheduler.core.components.collector import Collector
from scheduler.core.eventsqueue import InterruptionResolutionEvent, FaultResolutionEvent, WeatherClosureResolutionEvent, \
    InterruptionEvent, FaultEvent, WeatherClosureEvent
from scheduler.core.eventsqueue.nightchanges import NightlyTimeline
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
                                 collector: Collector) -> RunSummary:

        # scores_per_program: Dict[ProgramID, float] = {}
        metrics_per_program: Dict[ProgramID, float] = {}
        metrics_per_band: Dict[str, float] = {}
        programs = {}

        for night_idx in nights:
            # Setup for the night entire time losses
            timeline.time_losses.setdefault(night_idx, {})

            for site in sites:
                timeline_time_losses = {StatCalculator._FAULT_KEY: 0,
                                        StatCalculator._WEATHER_KEY: 0}
                timeline.time_losses[night_idx].setdefault(site, {})

                # Gather unsolved interruptions during the night.
                interruptions = []
                for entry in timeline.timeline[night_idx][site]:
                    if isinstance(entry.event, InterruptionEvent):
                        interruptions.append(entry.event)
                    elif isinstance(entry.event, InterruptionResolutionEvent):
                        if interruptions:
                            interruptions.pop()  # remove reported interruption and register the time loss
                        if isinstance(entry.event, FaultResolutionEvent):
                            timeline_time_losses[StatCalculator._FAULT_KEY] += int(entry.event.time_loss.total_seconds()/60)
                        elif isinstance(entry.event, WeatherClosureResolutionEvent):
                            timeline_time_losses[StatCalculator._WEATHER_KEY] += int(entry.event.time_loss.total_seconds()/60)

                # Unsolved interruptions for the night
                for e in interruptions:
                    if isinstance(e, FaultEvent):
                        time_loss = timeline.timeline[night_idx][site][-1].event.time - e.time
                        timeline_time_losses[StatCalculator._FAULT_KEY] += int(time_loss.total_seconds() / 60)

                    elif isinstance(e, WeatherClosureEvent):
                        time_loss = timeline.timeline[night_idx][site][-1].event.time - e.time
                        timeline_time_losses[StatCalculator._WEATHER_KEY] += int(time_loss.total_seconds() / 60)

                # Store the whole night time losses for the specified night and site
                timeline.time_losses[night_idx][site] = timeline_time_losses
                for entry in timeline.timeline[night_idx][site]:
                    # Save the time losses for the specific plan
                    time_losses = {StatCalculator._FAULT_KEY: 0,
                                   StatCalculator._WEATHER_KEY: 0,
                                   StatCalculator._UNSCHEDULE_KEY: 0}

                    # Some plans are shown empty if the telescope is closed.
                    if entry.plan_generated is None:
                        continue

                    plan = entry.plan_generated  # Update last plan

                    if 'Morning' in entry.event.description:
                        time_losses[StatCalculator._FAULT_KEY] = timeline_time_losses[StatCalculator._FAULT_KEY]
                        time_losses[StatCalculator._WEATHER_KEY] = timeline_time_losses[StatCalculator._WEATHER_KEY]

                        for v in plan.visits:
                            obs = collector.get_observation(v.obs_id)
                            program = collector.get_program(obs.belongs_to)

                            # Check if program is on the table
                            metrics_per_program.setdefault(program.id, 0.0)
                            metrics_per_band.setdefault(obs.band.name, 0.0)

                            # Calculate the metric in the program
                            metrics_per_program[program.id] += sum(v.metric)
                            metrics_per_band[obs.band.name] += sum(v.metric)

                    time_losses[StatCalculator._UNSCHEDULE_KEY] = (plan.time_left() -
                                                                   time_losses[StatCalculator._FAULT_KEY] -
                                                                   time_losses[StatCalculator._WEATHER_KEY])

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

                        completion_fraction[program.band] += 1

                        # Calculate altitude data
                        ti = collector.get_target_info(visit.obs_id)
                        end_time_slot = visit.start_time_slot + visit.time_slots
                        values = ti[night_idx].alt[visit.start_time_slot: end_time_slot]
                        alt_degs = [val.dms[0] + (val.dms[1] / 60) + (val.dms[2] / 3600) for val in values]
                        plan.alt_degs.append(alt_degs)

                    program_completion = {p.id: StatCalculator.calculate_program_completion(programs[p])
                                          for p in programs}
                    plan.night_stats = NightStats(time_losses,
                                                  plan_score,
                                                  n_toos,
                                                  completion_fraction,
                                                  program_completion)


        plans_summary: Summary = {}
        for p_id in metrics_per_program:
            program = collector.get_program(p_id)
            completion = StatCalculator.calculate_program_completion(program)

            metric = metrics_per_program[p_id]
            # score = scores_per_program[p_id]
            plans_summary[p_id.id] = (completion, metric)

        return RunSummary(plans_summary, metrics_per_band)

    @staticmethod
    def calculate_program_completion(program: Program) -> str:
        total_used = program.program_used()
        prog_total = program.program_awarded()
        return f'{float(total_used.total_seconds() / prog_total.total_seconds()) * 100:.1f}%'
