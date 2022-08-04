import strawberry
import pytz
from datetime import datetime
from typing import List
from app.common.minimodel import ObservationID, Site
from app.common.plans import Plan, Plans, Visit


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


@strawberry.input
class CreateNewScheduleInput:
    """
    Input for creating a new schedule.
    """
    start_time: str
    end_time: str
