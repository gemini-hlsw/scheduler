# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
The Visibility tab's four GraphQL fields, with the service layer mocked — these
assert the resolver contract, not the diffing logic (covered in
services/visibility_status).

The invariant worth guarding: only reference labels (``G-…``) cross the wire.
The ODB GIDs (``o-…`` / ``p-…``) are matching keys internal to the services, and
no field may expose one, so the UI cannot accidentally show an id an operator
cannot look up.
"""
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone

import pytest

from scheduler.graphql_mid import schema as schema_module
from scheduler.services.visibility_status.coverage import (
    CoverageSummary,
    GroupCoverage,
    ObservationCoverage,
    ObservationCoveragePage,
)
from scheduler.services.visibility_status.status import AggregatorStatus
from scheduler.services.visibility_status.tonight import (
    VisibleInterval,
    VisibleObservation,
    VisibleObservationsPage,
)

_NIGHT = date(2026, 7, 29)
_READ_AT = datetime(2026, 7, 29, 18, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def no_database(monkeypatch):
    """Resolvers open a session; CI has no DATABASE_URL."""

    @asynccontextmanager
    async def _scope():
        yield object()

    monkeypatch.setattr(schema_module, "session_scope", _scope)


@pytest.fixture
def stub_services(monkeypatch):
    def _install(name, value):
        async def _call(*args, **kwargs):
            return value
        monkeypatch.setattr(schema_module, name, _call)

    return _install


@pytest.mark.asyncio
async def test_visibility_coverage_reports_totals_and_breakdowns(
    scheduler_schema, stub_services
):
    stub_services("get_coverage_summary", CoverageSummary(
        expected=4, stored=2, pending=1, missing=1, skipped=0,
        per_program=[GroupCoverage(key="G-2026A-0001", expected=4, stored=2,
                                   pending=1, missing=1, skipped=0)],
        per_site=[GroupCoverage(key="GN", expected=4, stored=2, pending=1,
                                missing=1, skipped=0)],
        night_date=_NIGHT, odb_read_at=_READ_AT, pending_known=True,
    ))

    result = await scheduler_schema.execute("""
        query {
            visibilityCoverage {
                nightDate odbReadAt expected stored pending missing skipped
                isComplete pendingKnown
                perProgram { key expected missing }
                perSite { key expected missing }
            }
        }
    """)

    assert result.errors is None
    data = result.data["visibilityCoverage"]
    assert data["expected"] == 4
    assert data["missing"] == 1
    assert data["isComplete"] is False
    assert data["perProgram"][0]["key"] == "G-2026A-0001"
    assert data["perSite"][0]["key"] == "GN"


@pytest.mark.asyncio
async def test_complete_coverage_is_reported_as_complete(
    scheduler_schema, stub_services
):
    stub_services("get_coverage_summary", CoverageSummary(
        expected=2, stored=1, pending=0, missing=0, skipped=1,
        night_date=_NIGHT, odb_read_at=_READ_AT,
    ))

    result = await scheduler_schema.execute(
        "query { visibilityCoverage { isComplete skipped } }"
    )

    assert result.errors is None
    # Skipped observations can never be stored, so they must not block
    # "complete" — otherwise it is unreachable forever.
    assert result.data["visibilityCoverage"]["isComplete"] is True
    assert result.data["visibilityCoverage"]["skipped"] == 1


@pytest.mark.asyncio
async def test_observation_coverage_is_paginated_and_label_keyed(
    scheduler_schema, stub_services
):
    stub_services("list_observation_coverage", ObservationCoveragePage(
        observations=[
            ObservationCoverage(
                observation_id="G-2026A-0001-Q-0001",
                program_label="G-2026A-0001", site="GN", target_name="Vega",
                status="MISSING", skip_reason=None,
            ),
        ],
        total=137, night_date=_NIGHT, odb_read_at=_READ_AT,
    ))

    result = await scheduler_schema.execute("""
        query {
            observationCoverage(limit: 50, offset: 0, status: MISSING) {
                total
                nightDate
                observations {
                    observationId programLabel site targetName status skipReason
                }
            }
        }
    """)

    assert result.errors is None
    page = result.data["observationCoverage"]
    assert page["total"] == 137
    row = page["observations"][0]
    assert row["observationId"] == "G-2026A-0001-Q-0001"
    assert row["status"] == "MISSING"
    # The reference label, never the o- GID.
    assert row["observationId"].startswith("G-")


@pytest.mark.asyncio
async def test_observation_coverage_never_exposes_odb_gids(scheduler_schema):
    # A GID field would let the UI show an id operators cannot look up.
    result = await scheduler_schema.execute("""
        query { __type(name: "ObservationCoverage") { fields { name } } }
    """)

    assert result.errors is None
    names = {f["name"] for f in result.data["__type"]["fields"]}
    assert "internalId" not in names
    assert "programId" not in names


@pytest.mark.asyncio
async def test_visible_observations_returns_intervals_and_remaining_time(
    scheduler_schema, stub_services
):
    stub_services("list_visible_observations", VisibleObservationsPage(
        site="GN", night_date=_NIGHT, total=12, total_remaining_minutes=300,
        observations=[
            VisibleObservation(
                observation_id="G-2026A-0001-Q-0001", site="GN",
                target_name="Vega", night_date=_NIGHT,
                remaining_minutes=120, remaining_minutes_from_now=45,
                intervals=[VisibleInterval(
                    start=datetime(2026, 7, 29, 4, 0, tzinfo=timezone.utc),
                    end=datetime(2026, 7, 29, 6, 0, tzinfo=timezone.utc),
                )],
            ),
        ],
    ))

    result = await scheduler_schema.execute("""
        query {
            visibleObservations(site: "GN", limit: 50, offset: 0) {
                site nightDate total totalRemainingMinutes
                observations {
                    observationId targetName
                    remainingMinutes remainingMinutesFromNow
                    intervals { start end }
                }
            }
        }
    """)

    assert result.errors is None
    page = result.data["visibleObservations"]
    assert page["site"] == "GN"
    assert page["total"] == 12
    observation = page["observations"][0]
    assert observation["remainingMinutes"] == 120
    assert observation["remainingMinutesFromNow"] == 45
    assert len(observation["intervals"]) == 1


@pytest.mark.asyncio
async def test_aggregator_status_exposes_phase_and_eta(
    scheduler_schema, stub_services
):
    stub_services("get_typed_aggregator_status", AggregatorStatus(
        active=True, stale=False, holder="scheduler.1",
        started_at="2026-07-29T12:00:00+00:00", heartbeat_at=None,
        finished_at=None, phase="stage2", progress_current=40,
        progress_total=184, progress_unit="nights", elapsed_seconds=120.0,
        eta_seconds=432.0, detail='{"phase": "stage2"}',
    ))

    result = await scheduler_schema.execute("""
        query {
            visibilityAggregatorStatus {
                active stale holder phase
                progressCurrent progressTotal progressUnit
                elapsedSeconds etaSeconds
            }
        }
    """)

    assert result.errors is None
    status = result.data["visibilityAggregatorStatus"]
    assert status["active"] is True
    assert status["phase"] == "stage2"
    assert status["progressCurrent"] == 40
    assert status["etaSeconds"] == 432.0


@pytest.mark.asyncio
async def test_idle_aggregator_reports_null_progress(
    scheduler_schema, stub_services
):
    stub_services("get_typed_aggregator_status", AggregatorStatus(
        active=False, stale=False, holder=None, started_at=None,
        heartbeat_at=None, finished_at=None, phase=None,
        progress_current=None, progress_total=None, progress_unit=None,
        elapsed_seconds=None, eta_seconds=None, detail=None,
    ))

    result = await scheduler_schema.execute("""
        query {
            visibilityAggregatorStatus { active phase etaSeconds progressCurrent }
        }
    """)

    assert result.errors is None
    status = result.data["visibilityAggregatorStatus"]
    assert status["active"] is False
    assert status["phase"] is None
    assert status["etaSeconds"] is None
