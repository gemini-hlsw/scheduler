# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from astropy.time import Time
from lucupy.minimodel import Semester, Site
from typing import final, FrozenSet, Optional
from datetime import datetime

from .blueprint import CollectorBlueprint
from .schedulerbuilder import SchedulerBuilder
from scheduler.core.components.collector import Collector
from scheduler.core.sources.sources import Sources
from scheduler.core.programprovider.gpp import gpp_program_data, GppProgramProvider
from scheduler.core.statscalculator import StatCalculator
from scheduler.core.events.queue import EventQueue


__all__ = [
    'SimulationBuilder',
]

@final
class SimulationBuilder(SchedulerBuilder):
    """Simulation mode is used to predict future plans based on current GPP data

    Attributes:

    """

    def __init__(self, sources: Sources, events: EventQueue):
        super().__init__(sources, events)
        self.stats = StatCalculator

    def build_collector(self,
                        start: datetime,
                        end: datetime,
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
        collector.load_programs(
            program_provider_class=GppProgramProvider,
            data=gpp_program_data(program_list)
        )
        return collector


    async def async_build_collector(
        self,
        start: datetime,
        end: datetime,
        num_of_nights: int,
        sites: FrozenSet[Site],
        semesters: FrozenSet[Semester],
        blueprint: CollectorBlueprint,
        night_start_time: Time | None = None,
        night_end_time: Time | None = None,
        program_list: Optional[bytes] = None
    ) -> Collector:
        collector = super().build_collector(start,
                                            end,
                                            num_of_nights,
                                            sites,
                                            semesters,
                                            blueprint,
                                            night_start_time,
                                            night_end_time)
        async_data = await gpp_program_data(program_list)
        data = [item async for item in async_data]
        await collector.async_load_programs(
            program_provider_class=GppProgramProvider,
            data=data
        )
        return collector

    def _setup_event_queue(self,
                           start: datetime,
                           num_nights_to_schedule: int,
                           sites: FrozenSet[Site]) -> None:
        """
        Load all the events for the event queue from the different services for the number of nights to schedule.
        """
        for site in sites:
            ...
