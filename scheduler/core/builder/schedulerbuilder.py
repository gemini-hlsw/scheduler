# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import abstractmethod, ABC
from typing import FrozenSet, Optional, List

from astropy.time import Time
from lucupy.minimodel import Semester, Site

from .blueprint import CollectorBlueprint, SelectorBlueprint, OptimizerBlueprint
from scheduler.core.components.collector import Collector
from scheduler.core.components.selector import Selector
from scheduler.core.components.selector.timebuffer import create_time_buffer
from scheduler.core.components.optimizer import Optimizer
from scheduler.core.sources.sources import Sources
from scheduler.core.events.queue import EventQueue
from scheduler.core.storage_manager import storage_manager


__all__ = [
    'SchedulerBuilder',
]


class SchedulerBuilder(ABC):
    """
    Allows building different components individually and the general scheduler itself.
    """
    def __init__(self, sources: Sources, events: EventQueue):
        self.sources = sources
        self.events = events

        # TODO: DB storage?
        self.storage = None

    def build_collector(self,
                        start: Time,
                        end: Time,
                        num_of_nights: int,
                        sites: FrozenSet[Site],
                        semesters: FrozenSet[Semester],
                        blueprint: CollectorBlueprint,
                        programs_ids: List[str]) -> Collector:
        # TODO: Removing sources from Collector I think it was an idea
        # TODO: we might want to implement so all these are static methods.
        print('sites in builder: ', sites)
        programs = storage_manager.get_programs(programs_ids, sites)
        collector = Collector(
            start,
            end,
            num_of_nights,
            sites,
            semesters,
            self.sources,
            *blueprint,
            _programs=programs
        )
        return collector

    @staticmethod
    def build_selector(collector: Collector,
                       num_nights_to_schedule: int,
                       blueprint: SelectorBlueprint) -> Selector:
        return Selector(collector=collector,
                        num_nights_to_schedule=num_nights_to_schedule,
                        time_buffer=create_time_buffer(*blueprint))

    @staticmethod
    def build_optimizer(blueprint: OptimizerBlueprint) -> Optimizer:
        return Optimizer(algorithm=blueprint.algorithm)

    @abstractmethod
    def _setup_event_queue(self,
                           start: Time,
                           num_nights_to_schedule: int,
                           sites: FrozenSet[Site]) -> None:
        """
        Set up the event queue for this particular mode of the Scheduler.
        Note that we pass AstroPy times to this method because we will localize them for each site
        since data regarding events is typically provided in local time.
        """
        ...
