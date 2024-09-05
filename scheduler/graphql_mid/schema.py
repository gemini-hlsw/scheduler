# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
from typing import List, AsyncGenerator, Dict

import strawberry # noqa
from astropy.time import Time
from lucupy.minimodel.site import Site

from scheduler.core.components.ranker import RankerParameters
from scheduler.engine import Engine, SchedulerParameters
from scheduler.db.planmanager import PlanManager
from scheduler.services.logger_factory import create_logger
from scheduler.config import config

from .types import (SPlans, SNightTimelines, NewNightPlans, NightPlansError, NightPlansResponse, Version)
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
    def version(self) -> Version:
        return Version(version=config.app.version, changelog=config.app.changelog)

    @strawberry.field
    def plans(self) -> List[SPlans]:
        return PlanManager.get_plans()

    @strawberry.field
    def site_plans(self, site: Site) -> List[SPlans]:
        return [plans.for_site(site) for plans in PlanManager.get_plans()]

    @strawberry.field
    async def schedule(self, schedule_id: str, new_schedule_input: CreateNewScheduleInput) -> str:
        start = Time(new_schedule_input.start_time, format='iso', scale='utc')
        end = Time(new_schedule_input.end_time, format='iso', scale='utc')

        ranker_params = RankerParameters(new_schedule_input.thesis_factor,
                                         new_schedule_input.power,
                                         new_schedule_input.met_power,
                                         new_schedule_input.vis_power,
                                         new_schedule_input.wha_power)
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
