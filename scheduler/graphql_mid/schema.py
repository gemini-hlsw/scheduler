# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
import asyncio
from os import environ
from typing import Tuple, AsyncGenerator, Dict

import strawberry # noqa
from astropy.time import Time
from lucupy.minimodel.site import Site
from scheduler.context import schedule_id_var
from scheduler.core.components.ranker import RankerParameters
from scheduler.engine import Engine, SchedulerParameters
from scheduler.services.logger_factory import create_logger
from scheduler.config import config

from .types import (SPlans, SNightTimelines, NewNightPlans, NightPlansError, NightPlansResponse, Version, SRunSummary)
from .inputs import CreateNewScheduleInput


_logger = create_logger(__name__)


# TODO: All times need to be in UTC. This is done here but converted from the Optimizer plans, where it should be done.

def sync_schedule(params: SchedulerParameters, event: asyncio.Event) -> NewNightPlans:
    try:
        engine = Engine(params, event)
        plan_summary, timelines = engine.schedule()
        s_timelines = SNightTimelines.from_computed_timelines(timelines)
        s_plan_summary = SRunSummary.from_computed_run_summary(plan_summary)
        return NewNightPlans(night_plans=s_timelines, plans_summary=s_plan_summary)
    except RuntimeError as e:
        _logger.error(f'Error: {e}')
        return NightPlansResponse(error=NightPlansError(str(e)))


active_subscriptions: Dict[str, Tuple[asyncio.Queue, asyncio.Event]] = {}


@strawberry.type
class Query:

    @strawberry.field
    def version(self) -> Version:
        return Version(version=environ['APP_VERSION'], changelog=config.app.changelog)

    @strawberry.field
    async def schedule(self, schedule_id: str, new_schedule_input: CreateNewScheduleInput) -> str:
        schedule_id_var.set(schedule_id)

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
        _logger.info(f"Plan is on the queue! for: {schedule_id}\n{params}")

        if schedule_id not in active_subscriptions:
            queue = asyncio.Queue()
            event = asyncio.Event()
            active_subscriptions[schedule_id] = (queue, event)
        else:
            queue, event = active_subscriptions[schedule_id]

        event.set()
        task = asyncio.to_thread(sync_schedule, params, event)

        await queue.put(task)
        return f'Plan is on the queue! for {schedule_id}'

@strawberry.type
class Subscription:
    @strawberry.subscription
    async def queue_schedule(self, schedule_id: str) -> AsyncGenerator[NightPlansResponse, None]:
        try:
            if schedule_id not in active_subscriptions:
                queue = asyncio.Queue()
                event = asyncio.Event()
                active_subscriptions[schedule_id] = (queue, event)
            else:
                queue, event = active_subscriptions[schedule_id]
            try:
                while True:
                    try:
                        item = await queue.get()  # Wait for item from the queue
                        schedule_id_var.set(schedule_id)
                        _logger.info(f'Running ID: {schedule_id}')
                        result = await item
                        yield result  # Yield item to the subscription
                    except Exception as e:
                        _logger.error(f'Error: {e}')
                        yield NightPlansError(error=f'Error: {e}')
                        raise
            finally:
                if schedule_id in active_subscriptions:
                    del active_subscriptions[schedule_id]
        except asyncio.CancelledError:
            # Try to kill any running tasks clearing the event thread
            _logger.info(f'Connection of id {schedule_id} closed')
            event.clear()
