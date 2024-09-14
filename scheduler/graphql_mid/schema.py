# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
import os
import sys
from datetime import datetime
from typing import List, AsyncGenerator, Dict

import strawberry # noqa
from astropy.time import Time
from redis import asyncio as aioredis
from lucupy.minimodel.site import Site

from scheduler.core.components.ranker import RankerParameters
from scheduler.engine import Engine, SchedulerParameters
from scheduler.db.planmanager import PlanManager
from scheduler.services.logger_factory import create_logger

from .types import (SPlans, SNightTimelines, NewNightPlans, NightPlansError, NightPlansResponse)
from .inputs import CreateNewScheduleInput


_logger = create_logger(__name__)


# TODO: All times need to be in UTC. This is done here but converted from the Optimizer plans, where it should be done.

def sync_schedule(params: SchedulerParameters) -> NewNightPlans:
    engine = Engine(params)
    plan_summary, timelines = engine.run()

    s_timelines = SNightTimelines.from_computed_timelines(timelines)
    return NewNightPlans(night_plans=s_timelines, plans_summary=plan_summary)


active_subscriptions: Dict[str, asyncio.Queue] = {}


@strawberry.type
class Query:
    all_plans: List[SPlans] = strawberry.field(resolver=lambda: PlanManager.get_plans())

    @strawberry.field
    def plans(self) -> List[SPlans]:
        return PlanManager.get_plans()

    @strawberry.field
    def site_plans(self, site: Site) -> List[SPlans]:
        return [plans.for_site(site) for plans in PlanManager.get_plans()]

    @strawberry.field
    async def schedule(self,
                       new_schedule_input: CreateNewScheduleInput) -> NewNightPlans:
        try:
            start = Time(new_schedule_input.start_time, format='iso', scale='utc')
            end = Time(new_schedule_input.end_time, format='iso', scale='utc')

            ranker_params = RankerParameters(new_schedule_input.thesis_factor,
                                             new_schedule_input.power,
                                             new_schedule_input.met_power,
                                             new_schedule_input.vis_power,
                                             new_schedule_input.wha_power)
            #if new_schedule_input.program_file:
            #    program_file = (await new_schedule_input.program_file.read())
            #else:
            #    program_file = new_schedule_input.program_file

            params = SchedulerParameters(start, end,
                                         new_schedule_input.sites,
                                         new_schedule_input.mode,
                                         ranker_params,
                                         new_schedule_input.semester_visibility,
                                         new_schedule_input.num_nights_to_schedule)
            engine = Engine(params)
            plan_summary, timelines = engine.run()

            s_timelines = SNightTimelines.from_computed_timelines(timelines)

        except RuntimeError as e:
            raise RuntimeError(f'Schedule query error: {e}')
        return NewNightPlans(night_plans=s_timelines, plans_summary=plan_summary)

    @strawberry.field
    async def test_sub_query(self, schedule_id: str, new_schedule_input: CreateNewScheduleInput) -> str:
        start = Time(new_schedule_input.start_time, format='iso', scale='utc')
        end = Time(new_schedule_input.end_time, format='iso', scale='utc')

        ranker_params = RankerParameters(new_schedule_input.thesis_factor,
                                         new_schedule_input.power,
                                         new_schedule_input.met_power,
                                         new_schedule_input.vis_power,
                                         new_schedule_input.wha_power)
        file = None
        if new_schedule_input.program_file:
            file = await new_schedule_input.program_file.read()

        params = SchedulerParameters(start, end,
                                     new_schedule_input.sites,
                                     new_schedule_input.mode,
                                     ranker_params,
                                     new_schedule_input.semester_visibility,
                                     new_schedule_input.num_nights_to_schedule,
                                     file)

        _logger.info(f"Run ID: {schedule_id}\n{params}")

        task = asyncio.to_thread(sync_schedule, params)

        if schedule_id not in active_subscriptions:
            queue = asyncio.Queue()
            active_subscriptions[schedule_id] = queue
        else:
            queue = active_subscriptions[schedule_id]

        await queue.put(task)
        return f'Plan is on the queue! for {schedule_id}'


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def queue_schedule(self, schedule_id: str) -> AsyncGenerator[NightPlansResponse, None]:
        if schedule_id not in active_subscriptions:
            queue = asyncio.Queue()
            active_subscriptions[schedule_id] = queue
        else:
            queue = active_subscriptions[schedule_id]

        try:
            while True:
                try:
                    print(f'Queueing {schedule_id}')
                    item = await queue.get()  # Wait for item from the queue
                    print(f'Run ID: {schedule_id}')
                    result = await item
                    yield result  # Yield item to the subscription
                except Exception as e:
                    _logger.error(f'Error: {e}')
                    yield NightPlansError(error=f'Error: {e}')
                    raise
        finally:
            if schedule_id in active_subscriptions:
                del active_subscriptions[schedule_id]
