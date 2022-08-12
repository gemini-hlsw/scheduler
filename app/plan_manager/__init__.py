from common.meta import Singleton
from typing import List, NoReturn
from common.plans import Plans
from copy import deepcopy
from app.graphql.scalars import SPlans


class PlanManager(metaclass=Singleton):
    """
    A singleton class to store the current List[SPlans].
    1. The list represents the nights.
    2. The SPlans for each list entry is indexed by site to store the plan for the night.
    3. The SPlan is the plan for the site for the night, containing SVisits.
    """
    _plans: List[SPlans] = []

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
