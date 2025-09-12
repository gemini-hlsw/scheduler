# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from astropy.time import Time
from lucupy.minimodel import Semester, Site, ObservationStatus, Observation, QAState, TooType
from typing import final, FrozenSet, ClassVar, Iterable, Optional, Callable

from lucupy.types import ZeroTime

from .blueprint import CollectorBlueprint
from .schedulerbuilder import SchedulerBuilder
from scheduler.core.components.collector import Collector
from scheduler.core.sources.sources import Sources
from scheduler.core.programprovider.ocs import ocs_program_data, OcsProgramProvider
from scheduler.core.statscalculator import StatCalculator
from scheduler.core.events.queue import EventQueue


__all__ = [
    'ValidationBuilder',
]


@final
class ValidationBuilder(SchedulerBuilder):
    """Validation mode is used for validate the proper functioning

    Attributes:
        _obs_statuses_to_ready (ClassVar[FrozenSet[ObservationStatus]]):
            A set of statuses that show the observation is Ready.
    """

    # The default observations to set to READY in Validation mode.
    _obs_statuses_to_ready: ClassVar[FrozenSet[ObservationStatus]] = (
        frozenset([ObservationStatus.ONGOING, ObservationStatus.OBSERVED])
    )

    def __init__(self, sources: Sources, events: EventQueue):
        super().__init__(sources, events)
        self.stats = StatCalculator

    @staticmethod
    def _clear_observation_info(obs: Iterable[Observation],
                                obs_statuses_to_ready: FrozenSet[ObservationStatus] = _obs_statuses_to_ready,
                                observation_filter: Optional[Callable[[Observation], bool]] = None) -> None:
        """
        Given a single observation, clear the information associated with the observation.
        This is done when the Scheduler is run in Validation mode in order to start with a fresh observation.

        This consists of:
        1. Setting an observation status that is in obs_statuses_to_ready to READY (default: ONGOING or OBSERVED).
        2. Setting used times to 0 for the observation.

        Additional filtering may be done by specifying an optional filter for observations.
        """
        if observation_filter is not None:
            filtered_obs = (o for o in obs if observation_filter(o))
        else:
            filtered_obs = obs

        for o in filtered_obs:
            for atom in o.sequence:
                atom.program_used = ZeroTime
                atom.partner_used = ZeroTime
                atom.observed = False
                atom.qa_state = QAState.NONE

            if o.status in obs_statuses_to_ready:
                o.status = ObservationStatus.READY

            if o.too_type is not None:
                if o.too_type is TooType.RAPID:
                    o.status = ObservationStatus.ON_HOLD

    @staticmethod
    def reset_collector_observations(collector: Collector) -> None:
        """
        Clear out the observation information in the Collector by setting the times used to zero and setting
        the status of all observations to READY.
        """
        ValidationBuilder._clear_observation_info(
            collector.get_all_observations(),
            ValidationBuilder._obs_statuses_to_ready
        )

    def build_collector(self,
                        start: Time,
                        end: Time,
                        num_of_nights: int,
                        sites: FrozenSet[Site],
                        semesters: FrozenSet[Semester],
                        blueprint: CollectorBlueprint,
                        night_start_time: Time | None = None,
                        night_end_time: Time | None = None,
                        program_list: Optional[bytes] = None) -> Collector:

        collector = super().build_collector(start,
                                            end,
                                            num_of_nights,
                                            sites,
                                            semesters,
                                            blueprint,
                                            night_start_time,
                                            night_end_time)
        collector.load_programs(program_provider_class=OcsProgramProvider, data=ocs_program_data(program_list))
        ValidationBuilder.reset_collector_observations(collector)
        return collector

    def _setup_event_queue(self,
                           start: Time,
                           num_nights_to_schedule: int,
                           sites: FrozenSet[Site]) -> None:
        """
        Load all the events for the event queue from the different services for the number of nights to schedule.
        """
        for site in sites:
            ...
