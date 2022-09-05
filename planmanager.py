# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy
import shelve
from threading import Lock
from typing import List, NoReturn
from app.core.plans import Plans
from app.graphql.scalars import SPlans

from threading import Lock


class PlanManager:
    """
    A singleton class to store the current List[SPlans].
    1. The list represents the nights.
    2. The SPlans for each list entry is indexed by site to store the plan for the night.
    3. The SPlan is the plan for the site for the night, containing SVisits.
    """
    _plans: List[SPlans] = []
    _pm_lock: Lock = Lock()

    @staticmethod
    def get_plans() -> List[SPlans]:
        """
        Make a copy of the plans here and return them.
        This is to ensure that the plans are not corrupted after the
        lock is released.
        """
        # PlanManager._pm_lock.acquire()
        with shelve.open('plans', flag='r') as db: 
            plans = deepcopy(db['plans'])
        #PlanManager._pm_lock.release()
        return plans

    @staticmethod
    def set_plans(plans: List[Plans]) -> NoReturn:
        """
        Note that we are converting List[Plans] to List[SPlans].
        """
        # PlanManager._pm_lock.acquire()
        calculated_plans = deepcopy(plans)
        with shelve.open('plans') as db:
            db['plans'] = [ 
                            SPlans.from_computed_plans(p) for p in calculated_plans
                          ]
        
        # PlanManager._pm_lock.release()

#plan_manager = PlanManager()