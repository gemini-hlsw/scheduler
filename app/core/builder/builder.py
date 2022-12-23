# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from astropy.time import Time

from .blueprint import CollectorBlueprint, OptimizerBlueprint
from app.core.calculations import Selection
from app.core.components.collector import Collector
from app.core.components.selector import Selector
from app.core.components.optimizer import Optimizer
from app.core.scheduler.modes import dispatch_with
from app.config import config


@dispatch_with(config.mode)
class SchedulerBuilder:
    """Allows building different components individually and the general scheduler itself.

    """
    @staticmethod
    def build_collector(start: Time, end: Time, blueprint: CollectorBlueprint) -> Collector:
        return Collector(start, end, *blueprint)
    
    @staticmethod
    def build_selector(collector: Collector):
        return Selector(collector=collector)
    
    @staticmethod
    def build_optimizer(selection: Selection, blueprint: OptimizerBlueprint) -> Optimizer:
        return Optimizer(selection, algorithm=blueprint.algorithm)
