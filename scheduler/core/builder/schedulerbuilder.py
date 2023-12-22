# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC
from astropy.time import Time
from lucupy.minimodel import CloudCover, ImageQuality, Semester, Site
from typing import Dict, FrozenSet, Optional

from .blueprint import CollectorBlueprint, OptimizerBlueprint
from scheduler.core.components.collector import Collector
from scheduler.core.components.selector import Selector
from scheduler.core.components.optimizer import Optimizer
from scheduler.core.sources import Sources
from scheduler.core.eventsqueue import EventQueue


class SchedulerBuilder(ABC):
    """
    Allows building different components individually and the general scheduler itself.
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
                       cc_per_site: Optional[Dict[Site, CloudCover]] = None,
                       iq_per_site: Optional[Dict[Site, ImageQuality]] = None):
        return Selector(collector=collector,
                        num_nights_to_schedule=num_nights_to_schedule,
                        cc_per_site=cc_per_site or {},
                        iq_per_site=iq_per_site or {})

    @staticmethod
    def build_optimizer(blueprint: OptimizerBlueprint) -> Optimizer:
        return Optimizer(algorithm=blueprint.algorithm)
