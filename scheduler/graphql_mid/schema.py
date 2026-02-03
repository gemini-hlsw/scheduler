# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
from os import environ
from typing import AsyncGenerator, Dict
from datetime import datetime, UTC

import numpy as np
import strawberry # noqa
from astropy.coordinates import Angle
from astropy.time import Time
import astropy.units as u
from lucupy.minimodel import TimeslotIndex, NightIndex, VariantSnapshot, ImageQuality, CloudCover
from scheduler.context import schedule_id_var
from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.components.ranker import RankerParameters
from scheduler.engine import Engine, SchedulerParameters
from scheduler.server.process_manager import process_manager
from scheduler.services.logger_factory import create_logger
from scheduler.shared_queue import plan_response_queue

from .types import (SPlans, SNightTimelines, NewNightPlans, NightPlansError, Version, SRunSummary,
                    NewPlansRT, NightPlansResponseRT)
from .inputs import CreateNewScheduleInput, CreateNewScheduleRTInput
from ..core.plans import NightStats
from ..events import OnDemandScheduleEvent

_logger = create_logger(__name__)


# TODO: All times need to be in UTC. This is done here but converted from the Optimizer plans, where it should be done.

def sync_schedule(params: SchedulerParameters) -> NewNightPlans:
    engine = Engine(params)
    plan_summary, timelines = engine.schedule()
    s_timelines = SNightTimelines.from_computed_timelines(timelines)
    s_plan_summary = SRunSummary.from_computed_run_summary(plan_summary)
    return NewNightPlans(night_plans=s_timelines, plans_summary=s_plan_summary)

def sync_rt_schedule(params: SchedulerParameters, night_start_time: Time, night_end_time: Time, initial_variant: VariantSnapshot) -> NewPlansRT:
    engine = Engine(params, night_start_time=night_start_time, night_end_time=night_end_time)
    scp = engine.build()

    site = list(params.sites)[0]

    scp.selector.update_site_variant(site, initial_variant)
    plans = scp.run(site, np.array([NightIndex(0)]), TimeslotIndex(0))

    plans.plans[site].night_stats = NightStats({},0.0,0,{},{})
    plans.plans[site].alt_degs = []
    # Calculate altitude data
    for visit in plans.plans[site].visits:
        ti = scp.collector.get_target_info(visit.obs_id)
        end_time_slot = visit.start_time_slot + visit.time_slots
        values = ti[plans.night_idx].alt[visit.start_time_slot: end_time_slot]
        alt_degs = [val.dms[0] + (val.dms[1] / 60) + (val.dms[2] / 3600) for val in values]
        plans.plans[site].alt_degs.append(alt_degs)
    splans = SPlans.from_computed_plans(plans, params.sites)

    return NewPlansRT(night_plans=splans)

active_subscriptions: Dict[str, asyncio.Queue] = {}


@strawberry.type
class Query:

    @strawberry.field
    def version(self) -> Version:
        return Version(version=environ['APP_VERSION'], changelog=[])

    @strawberry.field
    async def schedule_rt(self, schedule_id: str, new_schedule_rt_input: CreateNewScheduleRTInput) -> str:
        schedule_id_var.set(schedule_id)
        start = datetime.fromisoformat(new_schedule_rt_input.start_time)
        end = datetime.fromisoformat(new_schedule_rt_input.end_time)

        ranker_params = RankerParameters(new_schedule_rt_input.thesis_factor,
                                         new_schedule_rt_input.power,
                                         new_schedule_rt_input.met_power,
                                         new_schedule_rt_input.vis_power,
                                         new_schedule_rt_input.wha_power,
                                         new_schedule_rt_input.air_power)
        
        programs_list = None
        if new_schedule_rt_input.programs:
            if len(new_schedule_rt_input.programs) > 0:
                programs_list = new_schedule_rt_input.programs

        params = SchedulerParameters(start,
                                     end,
                                     new_schedule_rt_input.sites,
                                     SchedulerModes.SIMULATION,
                                     ranker_params,
                                     False,
                                     1,
                                     programs_list)

        # Operation specific inputs
        # Get night start and end
        night_start = Time(new_schedule_rt_input.night_start_time, format='iso', scale='utc')
        night_end = Time(new_schedule_rt_input.night_end_time, format='iso', scale='utc')

        # Get new weathes inputs
        image_quality = ImageQuality(new_schedule_rt_input.image_quality)
        cloud_cover = CloudCover(new_schedule_rt_input.cloud_cover)
        wind_speed = new_schedule_rt_input.wind_speed
        wind_direction = new_schedule_rt_input.wind_direction

        initial_variant = VariantSnapshot(iq=image_quality,
                                          cc=cloud_cover,
                                          wind_dir=Angle(wind_direction, unit=u.deg),
                                          wind_spd=wind_speed * (u.m / u.s))

        task = asyncio.to_thread(sync_rt_schedule, params, night_start, night_end, initial_variant)
        if schedule_id not in active_subscriptions:
            queue = asyncio.Queue()
            active_subscriptions[schedule_id] = queue
        else:
            queue = active_subscriptions[schedule_id]

        await queue.put(task)
        return f'Plan is on the queue! for {schedule_id}'

    @strawberry.field
    async def schedule(self, schedule_id: str, new_schedule_input: CreateNewScheduleInput) -> str:
        schedule_id_var.set(schedule_id)

        start = datetime.fromisoformat(new_schedule_input.start_time)
        end = datetime.fromisoformat(new_schedule_input.end_time)

        ranker_params = RankerParameters(new_schedule_input.thesis_factor,
                                         new_schedule_input.power,
                                         new_schedule_input.met_power,
                                         new_schedule_input.vis_power,
                                         new_schedule_input.wha_power,
                                         new_schedule_input.air_power)
        programs_list = None
        if new_schedule_input.programs:
            if len(new_schedule_input.programs) > 0:
                programs_list = new_schedule_input.programs

        params = SchedulerParameters(start, end,
                                     new_schedule_input.sites,
                                     new_schedule_input.mode,
                                     ranker_params,
                                     new_schedule_input.semester_visibility,
                                     new_schedule_input.num_nights_to_schedule,
                                     programs_list)
        _logger.info(f"Plan is on the queue! for: {schedule_id}\n{params}")


        return f'Plan is on the queue! for {schedule_id}'

    @strawberry.field
    async def schedule_v2(self, new_schedule_rt_input: CreateNewScheduleRTInput)-> str:

        start = datetime.fromisoformat(new_schedule_rt_input.start_time)
        end = datetime.fromisoformat(new_schedule_rt_input.end_time)

        night_start = Time(new_schedule_rt_input.night_start_time, format='iso', scale='utc')
        night_end = Time(new_schedule_rt_input.night_end_time, format='iso', scale='utc')

        op_process = process_manager.get_operation_process()

        params = SchedulerParameters(
            start,
            end,
            programs_list=new_schedule_rt_input.programs,
            num_nights_to_schedule=1,
            semester_visibility=False
        )
        await op_process.update_params(params, night_start, night_end)

        utc_start = night_start.to_datetime(timezone=UTC)
        event = OnDemandScheduleEvent(
            description="On demand request",
            # time=datetime.now(UTC)
            time=utc_start,
        )
        await op_process.scheduler_queue.add_schedule_event(
            reason='On demand request',
            event=event,
        )
        return f'Plan is on the queue in the Operation Process!'



@strawberry.type
class Subscription:
    @strawberry.subscription
    async def queue_schedule(self, schedule_id: str) -> AsyncGenerator[NightPlansResponseRT, None]:
        if plan_response_queue.get(schedule_id) is None:
            plan_response_queue[schedule_id] = asyncio.Queue()
        queue = plan_response_queue[schedule_id]

        while True:
            try:
                _logger.info(f"Subscription: Waiting for plan response for {schedule_id}")
                result = await queue.get()  # Wait for item from the queue
                _logger.info(f"Subscription: Received plan response for {schedule_id}")
                _logger.debug(f'Result: {result}')
                yield result  # Yield item to the subscription
            except Exception as e:
                _logger.error(f'Error: {e}')
                yield NightPlansError(error=f'Error: {e}')
                raise
