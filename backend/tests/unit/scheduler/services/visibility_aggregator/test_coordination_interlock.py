# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

"""The aggregator interlock can be disabled for local runs (VIS_AGG_INTERLOCK)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from scheduler.services.visibility_aggregator import coordination


def _boom(*args, **kwargs):
    raise AssertionError("session_scope must not be used when the interlock is disabled")


def test_interlock_enabled_reads_config():
    with patch.object(coordination, "_agg_config", SimpleNamespace(interlock_enabled=False)):
        assert coordination._interlock_enabled() is False
    with patch.object(coordination, "_agg_config", SimpleNamespace(interlock_enabled=True)):
        assert coordination._interlock_enabled() is True
    # A missing key defaults to enabled (so prod behaviour is unchanged).
    with patch.object(coordination, "_agg_config", SimpleNamespace()):
        assert coordination._interlock_enabled() is True


@pytest.mark.asyncio
async def test_wait_until_idle_short_circuits_when_disabled():
    with patch.object(coordination, "_interlock_enabled", return_value=False), \
         patch.object(coordination, "session_scope", _boom):
        # Returns immediately as "idle" without touching the DB.
        assert await coordination.wait_until_aggregator_idle() is True


@pytest.mark.asyncio
async def test_signal_helpers_are_noops_when_disabled():
    with patch.object(coordination, "_interlock_enabled", return_value=False), \
         patch.object(coordination, "session_scope", _boom):
        # No DB access => no AssertionError raised.
        await coordination.signal_plan_in_progress()
        await coordination.signal_plan_done()
