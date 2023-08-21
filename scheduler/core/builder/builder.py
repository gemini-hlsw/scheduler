# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from astropy.time import Time
from lucupy.minimodel import CloudCover, ImageQuality, Semester, Site
from typing import FrozenSet

from .blueprint import CollectorBlueprint, OptimizerBlueprint
from scheduler.core.calculations import Selection
from scheduler.core.components.collector import Collector
from scheduler.core.components.selector import Selector
from scheduler.core.components.optimizer import Optimizer
from scheduler.core.service.modes import dispatch_with
from scheduler.core.sources import Sources, Origins
from scheduler.config import config


@dispatch_with(config.mode)
class SchedulerBuilder:
    """Allows building different components individually and the general scheduler itself.
    """
    sources = Sources()

    @staticmethod
    def build_collector(start: Time,
                        end: Time,
                        sites: FrozenSet[Site],
                        semesters: FrozenSet[Semester],
                        blueprint: CollectorBlueprint) -> Collector:
        return Collector(start, end, sites, semesters, SchedulerBuilder.sources, *blueprint)

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
