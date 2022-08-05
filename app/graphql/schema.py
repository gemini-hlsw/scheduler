import asyncio
import strawberry
from astropy.time import Time
from typing import List
from datetime import datetime

from app.process_manager import TaskType
from app.scheduler import Scheduler
from app.common.minimodel import Site
from app.process_manager import ProcessManager as PM
from app.plan_manager import PlanManager
from app.config import config
from .scalars import CreateNewScheduleInput, SPlans


# TODO: All times need to be in UTC. This is done here but converted from the Optimizer plans, where it should be done.


@strawberry.type
class Mutation:
    @strawberry.mutation
    async def new_schedule(self,
                           new_schedule_input: CreateNewScheduleInput) -> bool:
        try:
            start, end = Time(new_schedule_input.start_time, format='iso', scale='utc'), Time(new_schedule_input.end_time, format='iso', scale='utc')
        except ValueError:
            raise ValueError("Invalid time format. Must be ISO8601.")
        manager = PM()
        scheduler = Scheduler(start, end)
        manager.add_task(datetime.now(), scheduler, TaskType.STANDARD)
        await asyncio.sleep(10)
        return True


@strawberry.type
class Query:
    all_plans: List[SPlans] = strawberry.field(resolver=lambda: PlanManager.instance().get_plans())

    @strawberry.field
    def plans(self) -> List[SPlans]:
        return PlanManager.instance().get_plans()

    @strawberry.field
    def site_plans(self, site: Site) -> List[SPlans]:
        print(f'SITE IS {site}')
        return [plans.for_site(site) for plans in PlanManager.instance().get_plans()]
