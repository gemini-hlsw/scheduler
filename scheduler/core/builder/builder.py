# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from astropy.time import Time
from lucupy.minimodel import Site
from typing import FrozenSet

from .blueprint import CollectorBlueprint, OptimizerBlueprint
from scheduler.core.calculations import Selection
from scheduler.core.components.collector import Collector
from scheduler.core.components.selector import Selector
from scheduler.core.components.optimizer import Optimizer
from scheduler.core.service.modes import dispatch_with
from scheduler.config import config


@dispatch_with(config.mode)
class SchedulerBuilder:
    """Allows building different components individually and the general scheduler itself.
    """
    @staticmethod
<<<<<<< HEAD
    def build_collector(start: Time, end: Time, sites: FrozenSet[Site], blueprint: CollectorBlueprint) -> Collector:
        return Collector(start, end, sites, *blueprint)
    
=======
    # def build_collector(start: Time, end: Time, sites:FrozenSet[Site], blueprint: CollectorBlueprint) -> Collector:
    #     return Collector(start, end, sites, *blueprint)
    # *blueprint doesn't include the sites for some reason. Send each variable explicitly to include sites
    # and ensure the correct order since they are positional variables.
    def build_collector(start: Time, end: Time, blueprint: CollectorBlueprint) -> Collector:
        return Collector(start, end, blueprint.sites, blueprint.time_slot_length, blueprint.semesters,
                         blueprint.program_types, blueprint.obs_classes)

>>>>>>> 5114fbb (GreedyMax rebase, make build_collector and CollectorBlueprint consistent)
    @staticmethod
    def build_selector(collector: Collector):
        return Selector(collector=collector)
    
    @staticmethod
    def build_optimizer(selection: Selection, blueprint: OptimizerBlueprint) -> Optimizer:
        return Optimizer(selection, algorithm=blueprint.algorithm)
