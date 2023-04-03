# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from astropy.time import Time
from lucupy.minimodel import Semester, Site
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
    def build_collector(start: Time,
                        end: Time,
                        sites: FrozenSet[Site],
                        semesters: FrozenSet[Semester],
                        blueprint: CollectorBlueprint) -> Collector:
        return Collector(start, end, sites, semesters, *blueprint)

    @staticmethod
    def build_selector(collector: Collector):
        return Selector(collector=collector)

    @staticmethod
    def build_optimizer(selection: Selection, blueprint: OptimizerBlueprint) -> Optimizer:
        return Optimizer(selection, algorithm=blueprint.algorithm)
