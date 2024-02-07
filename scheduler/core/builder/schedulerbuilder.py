# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import abstractmethod, ABC

from astropy.time import Time
from lucupy.minimodel import CloudCover, ImageQuality, Semester, Site
from typing import Dict, FrozenSet, Optional

from .blueprint import CollectorBlueprint, SelectorBlueprint, OptimizerBlueprint
from scheduler.core.components.collector import Collector
from scheduler.core.components.selector import Selector
from scheduler.core.components.selector.timebuffer import create_time_buffer
from scheduler.core.components.optimizer import Optimizer
from scheduler.core.sources import Sources
from scheduler.core.eventsqueue import EventQueue


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
                        sites: FrozenSet[Site],
                        semesters: FrozenSet[Semester],
                        blueprint: CollectorBlueprint) -> Collector:
        # TODO: Removing sources from Collector I think it was an idea
        # TODO: we might want to implement so all these are static methods.
        collector = Collector(start, end, sites, semesters, self.sources, *blueprint)
        return collector

    @staticmethod
    def build_selector(collector: Collector,
                       num_nights_to_schedule: int,
                       blueprint: SelectorBlueprint,
                       cc_per_site: Optional[Dict[Site, CloudCover]] = None,
                       iq_per_site: Optional[Dict[Site, ImageQuality]] = None) -> Selector:
        return Selector(collector=collector,
                        num_nights_to_schedule=num_nights_to_schedule,
                        time_buffer=create_time_buffer(*blueprint),
                        cc_per_site=cc_per_site or {},
                        iq_per_site=iq_per_site or {})

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
