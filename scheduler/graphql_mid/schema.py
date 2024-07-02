# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
import os
from datetime import datetime
from typing import List, AsyncGenerator

import strawberry # noqa
from astropy.time import Time
from redis import asyncio as aioredis
from lucupy.minimodel.site import Site, ALL_SITES

from scheduler.core.sources.services import Services
from scheduler.core.sources.sources import Sources
from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.eventsqueue import EventQueue
from scheduler.core.components.ranker import RankerParameters
from scheduler.engine import Engine, SchedulerParameters
from scheduler.db.planmanager import PlanManager


from .types import (SPlans, NewNightPlans, ChangeOriginSuccess,
                    SourceFileHandlerResponse, SNightTimelines)
from .inputs import CreateNewScheduleInput, UseFilesSourceInput
from .scalars import SOrigin

REDIS_URL = os.environ.get("REDISCLOUD_URL")
redis = aioredis.from_url(REDIS_URL) if REDIS_URL else None

task_queue = asyncio.Queue()
# TODO: All times need to be in UTC. This is done here but converted from the Optimizer plans, where it should be done.


def synchronous_task():
    import time
    print('task')
    time.sleep(65)
    return 'new plan'


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
    async def test_redis(self) -> str:
        if redis:
            await redis.set("time_stamp", str(datetime.now().timestamp()))
            value = await redis.get("time_stamp")
            return value.decode()
        else:
            ValueError("REDISCLOUD_URL env var is not set up correctly.")

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
    async def test_sub_query(self) -> str:
        task = asyncio.to_thread(synchronous_task)
        await task_queue.put(task)
        return 'Plan is on the queue!'


@strawberry.type
class Subscription:
    @strawberry.subscription
    async def queue_schedule(self) -> AsyncGenerator[str, None]:
        while True:
            item = await task_queue.get()  # Wait for item from the queue
            result = await item
            yield result  # Yield item to the subscription
