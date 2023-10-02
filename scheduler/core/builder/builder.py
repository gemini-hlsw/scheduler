# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from astropy.time import Time
from lucupy.minimodel import CloudCover, ImageQuality, Semester, Site, ObservationStatus, Observation, QAState
from typing import FrozenSet, ClassVar, Iterable, Optional, Callable

from lucupy.types import ZeroTime

from .blueprint import CollectorBlueprint, OptimizerBlueprint
from scheduler.core.components.collector import Collector
from scheduler.core.components.selector import Selector
from scheduler.core.components.optimizer import Optimizer
from scheduler.core.sources import Sources
from scheduler.core.statscalculator import StatCalculator
from scheduler.core.eventsqueue import EventQueue


class SchedulerBuilder:
    """Allows building different components individually and the general scheduler itself.
    """
    def __init__(self, sources: Sources, events: EventQueue):
        self.sources = sources  # Services/Files/
        self.events = events  # EventManager() Emtpy by default
        self.storage = None  # DB storage

    def build_collector(self,
                        start: Time,
                        end: Time,
                        sites: FrozenSet[Site],
                        semesters: FrozenSet[Semester],
                        blueprint: CollectorBlueprint) -> Collector:
        # TODO: Removing sources from Collector I think it was an idea
        # TODO: we might want to implement so all these are static methods.
        return Collector(start, end, sites, semesters, self.sources, *blueprint)

    @staticmethod
    def build_selector(collector: Collector,
                       num_nights_to_schedule: int,
                       default_cc: CloudCover = CloudCover.CC50,
                       default_iq: ImageQuality = ImageQuality.IQ70):
        return Selector(collector=collector,
                        num_nights_to_schedule=num_nights_to_schedule,
                        default_cc=default_cc,
                        default_iq=default_iq)

    @staticmethod
    def build_optimizer(blueprint: OptimizerBlueprint) -> Optimizer:
        return Optimizer(algorithm=blueprint.algorithm)


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
        self.sim_manager = None  # This should bne called something else? Accountant?
        # Populate event manager, same as in Simulation.
        # EventManager.add(events)

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

    @staticmethod
    def update_collector(clean_collector: Collector) -> None:

        ValidationBuilder._clear_observation_info(
            clean_collector.get_all_observations(),
            ValidationBuilder._obs_statuses_to_ready
        )


class SimulationBuilder:
    pass


class OperationBuilder:
    pass
