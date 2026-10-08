# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
The memory guard stops an aggregator run between committed units of work once
the process is over its budget, so a big backfill is spread over several cron
ticks instead of swapping past the dyno quota or being killed with the
coordination row held.
"""
from contextlib import asynccontextmanager
from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from scheduler.services.visibility_aggregator import aggregator, memory_guard, runner
from scheduler.services.visibility_aggregator.memory_guard import (
    MemoryBudgetExceeded,
    MemoryGuard,
    process_memory_mb,
)

_PROC_STATUS = """Name:\tpython
VmPeak:\t 1200000 kB
VmRSS:\t  409600 kB
VmSwap:\t  102400 kB
Threads:\t4
"""


def _usage(*values):
    """Patch process memory to return ``values`` (MB) in order."""
    return patch.object(memory_guard, "process_memory_mb", side_effect=list(values))


# --- measuring --------------------------------------------------------------

def test_memory_counts_resident_and_swap():
    # Past the quota Heroku swaps the dyno, so resident alone would plateau.
    with patch("builtins.open", mock_open(read_data=_PROC_STATUS)):
        assert process_memory_mb() == 500.0


def test_memory_falls_back_to_peak_without_proc():
    usage = SimpleNamespace(ru_maxrss=300 * 1024 * 1024)
    with patch("builtins.open", side_effect=FileNotFoundError), \
         patch.object(memory_guard.resource, "getrusage", return_value=usage), \
         patch.object(memory_guard.sys, "platform", "darwin"):
        assert process_memory_mb() == 300.0


# --- the guard --------------------------------------------------------------

def test_check_returns_usage_under_budget():
    with _usage(400.0):
        assert MemoryGuard(460).check("Stage 1 batch 1") == 400.0


def test_check_raises_over_budget_with_phase_and_numbers():
    with _usage(470.0), pytest.raises(MemoryBudgetExceeded) as raised:
        MemoryGuard(460).check("Stage 1 batch 7")

    assert raised.value.phase == "Stage 1 batch 7"
    assert raised.value.used_mb == 470.0
    assert raised.value.budget_mb == 460.0
    assert "Stage 1 batch 7" in str(raised.value)


def test_no_budget_disables_the_guard():
    with _usage(10_000.0):
        assert MemoryGuard(None).check("Stage 2 night 2026-10-02") == 10_000.0


def test_budget_comes_from_config():
    with patch.object(
        memory_guard, "config",
        SimpleNamespace(visibility_aggregator=SimpleNamespace(memory_budget_mb=460)),
    ):
        assert MemoryGuard.from_config().budget_mb == 460.0
    # A missing key leaves the guard off rather than failing the run.
    with patch.object(
        memory_guard, "config",
        SimpleNamespace(visibility_aggregator=SimpleNamespace()),
    ):
        assert MemoryGuard.from_config().budget_mb is None


# --- Stage 2 stops only after committing ------------------------------------

@pytest.mark.asyncio
async def test_stage2_stops_after_the_committed_night_that_crossed_the_budget():
    nights = (date(2026, 10, 1), date(2026, 10, 3))
    requests = [SimpleNamespace(observation_id="G-2026B-0001-Q-0001")]
    calc = SimpleNamespace(
        visibility_repo=SimpleNamespace(
            get_stored_observation_ids_on_night=AsyncMock(return_value=set())
        ),
        store_visibility=AsyncMock(return_value={"stored": 1}),
        session=SimpleNamespace(commit=AsyncMock()),
    )

    with _usage(400.0, 470.0), pytest.raises(MemoryBudgetExceeded) as raised:
        await aggregator._store_missing_visibility(
            calc, requests, {"G-2026B-0001-Q-0001": nights}, None,
            "2026-10-02T00:00:00+00:00", MemoryGuard(460),
        )

    # Night 1 under budget, night 2 committed and then over: night 3 never runs.
    assert calc.store_visibility.await_count == 2
    assert calc.session.commit.await_count == 2
    assert raised.value.phase == "Stage 2 night 2026-10-02"


# --- the runner releases the coordination row --------------------------------

@pytest.mark.asyncio
async def test_runner_releases_the_row_and_fails_the_run_when_stopped():
    @asynccontextmanager
    async def _session():
        yield MagicMock()

    release = AsyncMock()
    stop = MemoryBudgetExceeded("Stage 1 batch 3", 470.0, 460.0)
    with patch.object(runner, "init_db_engine", AsyncMock()), \
         patch.object(runner, "dispose_engine", AsyncMock()), \
         patch.object(runner, "_install_signal_handlers"), \
         patch.object(runner, "is_night_in_progress", return_value=False), \
         patch.object(runner, "session_scope", _session), \
         patch.object(runner.coordination, "is_plan_in_progress", AsyncMock(return_value=False)), \
         patch.object(runner.coordination, "acquire_aggregator", AsyncMock(return_value=True)), \
         patch.object(runner.coordination, "heartbeat_aggregator", AsyncMock()), \
         patch.object(runner.coordination, "release_aggregator", release), \
         patch.object(runner, "run_aggregation", AsyncMock(side_effect=stop)):
        assert await runner._run() == 1

    release.assert_awaited_once()
