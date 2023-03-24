# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import os
from copy import deepcopy
from typing import List, NoReturn, FrozenSet
from lucupy.minimodel import Site

from scheduler.core.plans import Plans
from scheduler.graphql_mid.types import SPlans
from definitions import ROOT_DIR
from .dbmanager import DBManager

db = DBManager(os.path.join(ROOT_DIR, 'plans'))
DB_KEY = 'plans'


class PlanManager:
    """
    A singleton class to store the current List[SPlans].
    1. The list represents the nights.
    2. The SPlans for each list entry is indexed by site to store the plan for the night.
    3. The SPlan is the plan for the site for the night, containing SVisits.
    """

    @staticmethod
    def get_plans() -> List[SPlans]:
        """
        Make a copy of the plans here and return them.
        This is to ensure that the plans are not corrupted after the
        lock is released.
        """
        try:
            plans = deepcopy(db.read())
            return plans
        except KeyError:
            return None

    @staticmethod
    def get_plans_by_input(start_date: str, end_date: str, site: FrozenSet[Site]) ->  List[SPlans]:
        """
        A more specific way to get plans by the `CreateNewSchedule` input.
        """
        try:
            plans = deepcopy(db.read(start_date, end_date, site))
            return plans
        except KeyError:
            return []
    @staticmethod
    def set_plans(plans: List[Plans]) -> NoReturn:
        """
        Note that we are converting List[Plans] to List[SPlans].
        """
        try:
            calculated_plans = deepcopy(plans)
            db.write([
                SPlans.from_computed_plans(p) for p in calculated_plans
            ])
        except KeyError:
            raise KeyError(f'Error on read!')
