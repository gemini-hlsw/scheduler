# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
from os import environ
from typing import AsyncGenerator, Dict
from datetime import datetime, UTC
from zoneinfo import ZoneInfo

import numpy as np
import strawberry # noqa
from astropy.coordinates import Angle
from astropy.time import Time
import astropy.units as u
from gpp_client.api import WhereProgram, WhereEqProposalStatus, ProposalStatus
from lucupy.minimodel import TimeslotIndex, NightIndex, VariantSnapshot, ImageQuality, CloudCover
from pydantic import ValidationError

from scheduler.context import schedule_id_var
from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.components.ranker import RankerParameters
from scheduler.engine import Engine, SchedulerParameters
from scheduler.orchestration import process_manager
from scheduler.services.logger_factory import create_logger
from scheduler.shared_queue import plan_response_subscribers
from scheduler.clients.scheduler_gpp_client import gpp_client_instance

from .types import (SPlans, SNightTimelines, NewNightPlans, NightPlansError, Version, SRunSummary,
                    NewPlansRT, NightPlansResponseRT, BuildParametersInput)
from .inputs import CreateNewScheduleInput, CreateNewScheduleRTInput

from ..core.plans import NightStats
from ..engine.params import build_params_store
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
    async def schedule(self, schedule_id: str, new_schedule_input: CreateNewScheduleInput) -> str:
        schedule_id_var.set(schedule_id)

        start = datetime.fromisoformat(new_schedule_input.start_time).replace(tzinfo=ZoneInfo("UTC"))
        end = datetime.fromisoformat(new_schedule_input.end_time).replace(tzinfo=ZoneInfo("UTC"))

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

        task = asyncio.to_thread(sync_schedule, params)

        if schedule_id not in plan_response_subscribers:
            plan_response_subscribers[schedule_id] = set()
            client_queue = asyncio.Queue()
            plan_response_subscribers[schedule_id].add(client_queue)
            queues = plan_response_subscribers[schedule_id]
        else:
            queues = plan_response_subscribers[schedule_id]

        # Is only one queue anyway as the process is subscribed by only one client.
        for q in queues:
            await q.put(task)
        _logger.info(f"Plan is on the queue! for: {schedule_id}\n{params}")
        return f'Plan is on the queue! for {schedule_id}'

    @strawberry.field
    async def schedule_v2(self, new_schedule_rt_input: CreateNewScheduleRTInput)-> str:

        night_start = Time(new_schedule_rt_input.night_start_time, format='iso', scale='utc')
        op_process = process_manager.get_operation_process()

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

    @strawberry.field
    async def available_programs(self)-> list[str]:
        client = gpp_client_instance.client
        where = WhereProgram(proposal_status=WhereEqProposalStatus(
            eq=ProposalStatus.ACCEPTED
        ))
        response = await client.program.get_all(where=where)
        ids = [p["id"] for p in response['matches']]
        return ids


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def queue_schedule(self, schedule_id: str) -> AsyncGenerator[NightPlansResponseRT, None]:
        if schedule_id not in plan_response_subscribers:
            plan_response_subscribers[schedule_id] = set()

        client_queue = asyncio.Queue()
        plan_response_subscribers[schedule_id].add(client_queue)

        try:
            while True:
                try:
                    _logger.info(f"Subscription: Waiting for plan response for {schedule_id}")
                    result = await client_queue.get()  # Wait for item from the queue
                    _logger.info(f"Subscription: Received plan response for {schedule_id}")
                    _logger.debug(f'Result: {result}')
                    yield result  # Yield item to the subscription
                except Exception as e:
                    _logger.error(f'Error: {e}')
                    yield NightPlansError(error=f'Error: {e}')
                    raise
        finally:
            plan_response_subscribers[schedule_id].discard(client_queue)
            if not plan_response_subscribers[schedule_id]:
                del plan_response_subscribers[schedule_id]


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def update_build_params(self, build_params_input: BuildParametersInput) -> str:
        msg = ''
        try:
            build_params = build_params_input.to_pydantic()
            await build_params_store.set(build_params)
            msg += f'Build Parameters updated successful'
        except ValidationError as e:
            msg+=f'Error: {e.errors()}'
        return msg
