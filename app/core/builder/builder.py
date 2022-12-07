# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Iterable

from astropy.time import Time
from lucupy.minimodel import Program

from .blueprint import Blueprint
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
    def build_collector(start: Time, end: Time, blueprint: Blueprint) -> Collector:
        return Collector(start, end, *blueprint)
    
    @staticmethod
    def build_selector(collector: Collector):
        return Selector(collector=collector)
    
    @staticmethod
    def build_optimizer(selection: Iterable[Program], blueprint: Blueprint) -> Selector:
        return Optimizer(selection, algorithm=blueprint.algorithm)

