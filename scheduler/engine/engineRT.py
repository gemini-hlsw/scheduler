# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from astropy.time import Time

from .params import SchedulerParameters
from scheduler.core.scp.scp import SCP

from scheduler.core.builder.modes import dispatch_with
from scheduler.core.builder import Blueprints
from scheduler.core.sources import Sources
from scheduler.core.plans import NightStats
from scheduler.services import logger_factory

from scheduler.graphql_mid.types import SPlans, NewPlansRT

from lucupy.minimodel import VariantSnapshot, ImageQuality, CloudCover
from astropy.coordinates import Angle
from astropy import units as u

from scheduler.core.events.queue import Event

from time import time


__all__ = [
    'EngineRT'
]

from ..core.components.ranker import DefaultRanker

_logger = logger_factory.create_logger(__name__)


class EngineRT:

    def __init__(self, params: SchedulerParameters, night_start_time: Time | None = None, night_end_time: Time | None = None):
        """
        Initializes the EngineRT with the given parameters.
        
        Args:
            params (SchedulerParameters): Parameters for the scheduler.
            night_start_time (Time | None): Optional start time of the night.
            night_end_time (Time | None): Optional end time of the night.
        """
        self.params = params
        self.sources = Sources()
        self.change_monitor = None
        self.start_time = time()
        self.night_start_time = night_start_time
        self.night_end_time = night_end_time

        self.build()
        self.init_variant()

    def build(self) -> SCP:
        """
        Creates a Scheduler Core Pipeline based on the parameters.
        Also initialize both the Event Queue , both needed for the scheduling process.
        """

        # Create builder based in the mode to create SCP
        builder = dispatch_with(self.sources, self.queue)

        collector = builder.build_collector(start=self.params.start,
                                            end=self.params.end_vis,
                                            num_of_nights=self.params.num_nights_to_schedule,
                                            sites=self.params.sites,
                                            semesters=self.params.semesters,
                                            blueprint=Blueprints.collector,
                                            night_start_time=self.night_start_time,
                                            night_end_time=self.night_end_time,
                                            program_list=self.params.programs_list)

        selector = builder.build_selector(collector=collector,
                                          num_nights_to_schedule=self.params.num_nights_to_schedule,
                                          blueprint=Blueprints.selector)

        optimizer = builder.build_optimizer(Blueprints.optimizer)
        ranker = DefaultRanker(collector,
                               self.params.night_indices,
                               self.params.sites,
                               params=self.params.ranker_parameters)

        self.scp = SCP(collector, selector, optimizer, ranker)

    def init_variant(self):
        # Should get variants from weather service in the future
        initial_variant = VariantSnapshot(iq=ImageQuality(1.0),
                                          cc=CloudCover(1.0),
                                          wind_dir=Angle(0.0, unit=u.deg),
                                          wind_spd=0.0 * (u.m / u.s))
        for site in self.params.sites:
            self.scp.selector.update_site_variant(site, initial_variant)

    def compute_event_plan(self, event: Event):
        # Get the timeslots associated with the sites with format
        # {site: {0: current_timeslot}}
        start_timeslot = {}
        for site in self.params.sites:
            night_start_time = self.scp.collector.night_events[site].times[0]
            event_timeslot = event.to_timeslot_index(night_start_time, self.scp.collector.time_slot_length)
            start_timeslot[site] = {0: event_timeslot}

        plans = self.scp.run_rt(start_timeslot)

        for site in self.params.sites:
            plans.plans[site].night_stats = NightStats({},0.0,0,{},{})
            plans.plans[site].alt_degs = []
            # Calculate altitude data
            for visit in plans.plans[site].visits:
                ti = self.scp.collector.get_target_info(visit.obs_id)
                end_time_slot = visit.start_time_slot + visit.time_slots
                values = ti[plans.night_idx].alt[visit.start_time_slot: end_time_slot]
                alt_degs = [val.dms[0] + (val.dms[1] / 60) + (val.dms[2] / 3600) for val in values]
                plans.plans[site].alt_degs.append(alt_degs)
            splans = SPlans.from_computed_plans(plans, self.params.sites)

        return NewPlansRT(night_plans=splans)
