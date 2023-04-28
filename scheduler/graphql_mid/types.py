# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import json
from datetime import datetime
from typing import List, Union, FrozenSet

import pytz
import strawberry  # noqa
from strawberry.scalars import JSON
from lucupy.minimodel import ObservationID, Site, ALL_SITES

from scheduler.core.plans import Plan, Plans, Visit, NightStats
from scheduler.graphql_mid.scalars import SObservationID


@strawberry.type
class SNightStats:
    """Night stats to display in the UI
    """
    timeloss: str
    plan_score: float
    plan_conditions: JSON
    n_toos: int
    completion_fraction: JSON

    @staticmethod
    def from_computed_night_stats(ns: NightStats) -> 'SNightStats':
        conditions = {'cc': ns.plan_conditions.cc,
                      'iq': ns.plan_conditions.iq,
                      'wv': ns.plan_conditions.wv,
                      'sb': ns.plan_conditions.sb}
        conditions = json.dumps(conditions)
        cf = json.dumps(ns.completion_fraction)
        return SNightStats(timeloss=ns.timeloss,
                           plan_score=ns.plan_score,
                           plan_conditions=conditions,
                           n_toos=ns.n_toos,
                           completion_fraction=cf)


@strawberry.type
class SVisit:
    """
    Represents a visit as part of a nightly Plan at a Site.
    """
    start_time: datetime
    obs_id: SObservationID
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
    site: strawberry.enum(Site)
    start_time: datetime
    end_time: datetime
    visits: List[SVisit]
    night_stats: SNightStats

    @staticmethod
    def from_computed_plan(plan: Plan) -> 'SPlan':
        return SPlan(
            site=plan.site,
            start_time=plan.start.astimezone(pytz.UTC),
            end_time=plan.end.astimezone(pytz.UTC),
            visits=[SVisit.from_computed_visit(visit) for visit in plan.visits],
            night_stats=SNightStats.from_computed_night_stats(plan.night_stats)
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
    def from_computed_plans(plans: Plans, sites: FrozenSet[Site]) -> 'SPlans':
        return SPlans(
            night_idx=plans.night,
            plans_per_site=[SPlan.from_computed_plan(plans[site]) for site in sites])

    def for_site(self, site: Site) -> 'SPlans':
        return SPlans(
            night_idx=self.night_idx,
            plans_per_site=[plans for plans in self.plans_per_site if plans.site == site])

@strawberry.type
class NewNightPlans:
    night_plans: List[SPlans]
    plans_summary: JSON


@strawberry.type
class NewScheduleSuccess:
    """
    Success response for creating a new schedule.
    """
    success: bool


@strawberry.type
class NewScheduleError:
    """
    Error response for creating a new schedule.
    """
    error: str


NewScheduleResponse = strawberry.union("NewScheduleResponse", types=(NewScheduleSuccess, NewScheduleError))  # noqa
