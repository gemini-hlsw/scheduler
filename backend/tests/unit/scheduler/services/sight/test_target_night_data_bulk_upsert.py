# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
Stage 1 bulk_upsert must compile to one INSERT ... ON CONFLICT DO UPDATE with
no RETURNING (so asyncpg pipelines it), overwrite every array column so a
stale row is refreshed in place, and chunk rows to bound bind buffers.
"""
import asyncio
from datetime import date, datetime, timezone

from sqlalchemy.dialects import postgresql

from scheduler.services.sight.database.repositories.target_night_data import (
    BULK_UPSERT_CHUNK,
    TargetNightDataRepository,
)


class _RecordingSession:
    """Captures execute() calls without touching a database."""

    def __init__(self):
        self.calls = []

    async def execute(self, stmt, params=None):
        self.calls.append((stmt, params))


def _row(i: int) -> dict:
    array = b"\x00" * 8 * 600
    return {
        'target_id': i,
        'site_id': 1,
        'night_date': date(2026, 10, 1),
        'night_duration_minutes': 600,
        'ra': array, 'dec': array, 'alt': array, 'az': array,
        'hourangle': array, 'airmass': array, 'par_ang': None,
        'target_updated_at': datetime(2026, 10, 1, tzinfo=timezone.utc),
    }


def test_bulk_upsert_statement_shape_and_chunking():
    session = _RecordingSession()
    repo = TargetNightDataRepository(session)

    n_rows = BULK_UPSERT_CHUNK * 2 + 7
    stored = asyncio.run(repo.bulk_upsert([_row(i) for i in range(n_rows)]))

    assert stored == n_rows
    assert [len(params) for _, params in session.calls] == [
        BULK_UPSERT_CHUNK, BULK_UPSERT_CHUNK, 7
    ]

    sql = str(session.calls[0][0].compile(dialect=postgresql.dialect()))
    assert 'ON CONFLICT ON CONSTRAINT uq_target_night_data_target_site_night DO UPDATE SET' in sql
    assert 'RETURNING' not in sql
    # A stale row is refreshed in place: every array and the staleness stamp.
    for column in ('alt', 'airmass', 'par_ang', 'target_updated_at', 'computed_at'):
        assert f'{column} = ' in sql.split('DO UPDATE SET', 1)[1]


def test_bulk_upsert_empty_rows_issues_no_queries():
    session = _RecordingSession()
    repo = TargetNightDataRepository(session)

    assert asyncio.run(repo.bulk_upsert([])) == 0
    assert session.calls == []
