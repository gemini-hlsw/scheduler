# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio

from gpp_client.api.custom_fields import VisitFields

from scheduler.core.meta import AsyncSingleton


class LastPlanMock:
    visits = []

    def get_observation(self, observationId):
        return MockObservation

    def current_visit(self):
        """Pointer to the current visit. Gets updated when a new visit is executed"""
        return VisitFields

    def resources(self):
        return []


class PlanDBClient(metaclass=AsyncSingleton):


    def __init__(self):
        # In reality this should call a database that would store this
        # but for testing this would just store from
        self._last_plan_state = None
        self._lock = asyncio.Lock()

    def get_last_plan(self) -> LastPlanMock:
        with self._lock:
            return self._last_plan_state

    def set_last_plan(self, last_plan: LastPlanMock):
        with self._lock:
            self._last_plan_state = last_plan
