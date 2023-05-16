# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import json
from datetime import datetime
from typing import List
import strawberry # noqa
from astropy.time import Time
from lucupy.minimodel import Site

from scheduler.core.builder import SchedulerBuilder
from scheduler.core.service.service import build_scheduler
from scheduler.core.sources import Origins
from scheduler.process_manager import setup_manager, TaskType
from scheduler.db.planmanager import PlanManager


from .types import (SPlans, NewScheduleResponse,
                    NewScheduleError, NewScheduleSuccess,
                    NewNightPlans, ChangeOriginSuccess)
from .inputs import CreateNewScheduleInput


builder = SchedulerBuilder()
builder.sources.set_origin(Origins.OCS) # Default now is OCS.


# TODO: All times need to be in UTC. This is done here but converted from the Optimizer plans, where it should be done.
@strawberry.type
class Mutation:
    '''
    @strawberry.mutation
    def change_mode():
        pass

    '''

    @strawberry.mutation
    def change_origin(new_origin: str) -> ChangeOriginSuccess:
        old = builder.sources.origin.value
        origin = Origins[new_origin]
        builder.sources.set_origin(origin)
        return ChangeOriginSuccess(old, new_origin)



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
    def schedule(self, new_schedule_input: CreateNewScheduleInput) -> NewNightPlans:
        plans = PlanManager.get_plans_by_input(new_schedule_input.start_time,
                                               new_schedule_input.end_time,
                                               new_schedule_input.site)
        plans_summary = {}
        if not plans:
            start, end = Time(new_schedule_input.start_time, format='iso', scale='utc'), \
                    Time(new_schedule_input.end_time, format='iso', scale='utc')
            scheduler = build_scheduler(start, end, new_schedule_input.site, builder)
            plans, plans_summary = scheduler()
        splans = [SPlans.from_computed_plans(p, new_schedule_input.site) for p in plans]
        # json_summary = json.dumps(plans_summary)
        return NewNightPlans(night_plans=splans, plans_summary=plans_summary)
