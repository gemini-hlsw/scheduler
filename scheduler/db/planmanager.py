# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy
from typing import List, NoReturn

from scheduler.core.plans import Plans
from scheduler.graphql_mid.scalars import SPlans
from definitions import ROOT_DIR
from .dbmanager import DBManager

db = DBManager(f'{ROOT_DIR}/plans')
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
        except Exception as err:
            raise Exception(f'Error on read: {err}')

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
        except Exception as err:
            raise Exception(f'Error on write: {err}')
