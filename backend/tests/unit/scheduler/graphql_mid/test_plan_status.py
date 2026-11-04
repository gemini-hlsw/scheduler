# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""The plan status reaches every UI: the current one on subscribe, then every change."""

import asyncio
from datetime import datetime, UTC
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scheduler.core.events.queue.events import OnDemandScheduleEvent
from scheduler.core.events.queue.plan_status import PlanStatusTracker
from scheduler.graphql_mid.server import schema

_SUB = """
subscription {
  planStatusUpdates { state eventDescription requestedByUser lastError lastDurationSeconds }
}
"""


@pytest.mark.asyncio
async def test_subscription_sends_current_status_then_changes():
    tracker = PlanStatusTracker()
    process = SimpleNamespace(plan_status=tracker)
    event = OnDemandScheduleEvent(
        site=None, time=datetime.now(UTC), description="From another browser", requested_by_user=True
    )

    with patch("scheduler.graphql_mid.schema.process_manager.get_operation_process", return_value=process):
        sub = await schema.subscribe(_SUB)
        first = await asyncio.wait_for(sub.__anext__(), timeout=2)
        assert first.errors is None
        assert first.data["planStatusUpdates"]["state"] == "IDLE"

        tracker.computing(event)
        computing = (await asyncio.wait_for(sub.__anext__(), timeout=2)).data["planStatusUpdates"]
        assert computing["state"] == "COMPUTING"
        assert computing["eventDescription"] == "From another browser"
        assert computing["requestedByUser"] is True

        tracker.finished("boom")
        done = (await asyncio.wait_for(sub.__anext__(), timeout=2)).data["planStatusUpdates"]
        assert done["state"] == "IDLE"
        assert done["lastError"] == "boom"
        assert done["lastDurationSeconds"] >= 0
        await sub.aclose()


@pytest.mark.asyncio
async def test_query_reads_the_current_status():
    tracker = PlanStatusTracker()
    process = SimpleNamespace(plan_status=tracker)
    with patch("scheduler.graphql_mid.schema.process_manager.get_operation_process", return_value=process):
        result = await schema.execute("{ planStatus { state } }")
    assert result.errors is None
    assert result.data == {"planStatus": {"state": "IDLE"}}
