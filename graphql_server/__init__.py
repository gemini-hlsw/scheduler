from copy import deepcopy
from datetime import date, datetime
from threading import Lock
from typing import List, NoReturn
import pytz

import strawberry
import uvicorn
from fastapi import FastAPI
from strawberry.asgi import GraphQL

from common.minimodel import ObservationID, Site
from common.plans import Plan, Plans, Visit

# NOTE: All times are in UTC.

# Hierarchy:
# List[Plans]: one entry for each night
#   Plans: for a given night, indexed by Site to get Plan
#     Plan: for a given site, a list of Visits and night information
#       Visit: One visit as part of a Plan


# TODO: We might want to refactor with common.plans to share code when possible.
# Strawberry classes and converters.

@strawberry.type
class SVisit:
    """
    Represents a visit as part of a nightly Plan at a Site.
    """
    start_time: datetime
    obs_id: ObservationID
    atom_start_idx: int
    atom_end_idx: int

    @staticmethod
    def from_computed_visit(visit: Visit) -> 'SVisit':
        return SVisit(start_time=visit.start_time.astimezone(pytz.UTC),
                      obs_id=visit.obs_id,
                      atom_start_idx=visit.atom_start_idx,
                      atom_end_idx=visit.atom_end_idx)


@strawberry.type
class SPlan:
    """
    A nightly Plan for a specific site.
    """
    site: Site
    start_time: datetime
    end_time: datetime
    visits: List[SVisit]

    @staticmethod
    def from_computed_plan(plan: Plan) -> 'SPlan':
        return SPlan(
            site=plan.site,
            start_time=plan.start.astimezone(pytz.UTC),
            end_time=plan.end.astimezone(pytz.UTC),
            visits=[SVisit.from_computed_visit(visit) for visit in plan.visits]
        )


@strawberry.type
class SPlans:
    """
    For a given night, a collection of Plan for each Site.
    """
    # TODO: Change this to date in UTC
    night_idx: int
    plans_per_site: List[SPlan]

    @staticmethod
    def from_computed_plans(plans: Plans) -> 'SPlans':
        return SPlans(
            night_idx=plans.night,
            plans_per_site=[SPlan.from_computed_plan(plans[site]) for site in Site]
        )

    def for_site(self, site: Site) -> 'SPlans':
        return SPlans(
            night_idx=self.night_idx,
            plans_per_site=[plans for plans in self.plans_per_site if plans.site == site]
        )


class PlanManager:
    """
    A singleton class to store the current List[SPlans].
    1. The list represents the nights.
    2. The SPlans for each list entry is indexed by site to store the plan for the night.
    3. The SPlan is the plan for the site for the night, containing SVisits.
    """
    _lock = Lock()
    _plans: List[SPlans] = []

    def __new__(cls):
        if not hasattr(cls, 'inst'):
            cls.inst = super(PlanManager, cls).__new__(cls)
            # cls.inst.a = 5
        return cls.inst

    @staticmethod
    def instance() -> 'PlanManager':
        return PlanManager()

    @staticmethod
    def get_plans() -> List[SPlans]:
        """
        Make a copy of the plans here and return them.
        This is to ensure that the plans are not corrupted after the
        lock is released.
        """
        PlanManager._lock.acquire()
        plans = deepcopy(PlanManager._plans)
        PlanManager._lock.release()
        return plans

    @staticmethod
    def set_plans(plans: List[Plans]) -> NoReturn:
        """
        Note that we are converting List[Plans] to List[SPlans].
        """
        PlanManager._lock.acquire()
        calculated_plans = deepcopy(plans)
        PlanManager._plans = [
            SPlans.from_computed_plans(p) for p in calculated_plans
        ]
        PlanManager._lock.release()


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


schema = strawberry.Schema(query=Query)
# graphql_app = GraphQLRouter(schema)
graphql_app = GraphQL(schema)
app = FastAPI()
# app.include_router(graphql_app, prefix='/graphql')
app.add_route('/graphql', graphql_app)
app.add_websocket_route('/graphql', graphql_app)


def start_graphql_server():
    # uvicorn.run('graphqlserver:app', reload=True, host='127.0.0.1', port=8000)
    uvicorn.run('graphql_server:app', host='127.0.0.1', port=8000)
