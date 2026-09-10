# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
bulk_upsert must compile to a single PostgreSQL INSERT ... ON CONFLICT DO
UPDATE with no RETURNING (RETURNING would disable asyncpg's executemany
pipelining and fall back to row-at-a-time statements), and must chunk the
submitted rows so bind buffers stay bounded.
"""
import asyncio
from datetime import date

from sqlalchemy.dialects import postgresql

from scheduler.services.sight.database.repositories.visibility_data import (
    BULK_UPSERT_CHUNK,
    VisibilityDataRepository,
)


class _RecordingSession:
    """Captures execute() calls without touching a database."""

    def __init__(self):
        self.calls = []

    async def execute(self, stmt, params=None):
        self.calls.append((stmt, params))


def _row(i: int) -> dict:
    return {
        'observation_id': f'GN-2018B-Q-1-{i}',
        'target_id': 1,
        'site_id': 1,
        'night_date': date(2018, 8, 1),
        'remaining_minutes': 42,
        'visible_ranges': [[0, 10]],
        'constraints': {},
    }


def test_bulk_upsert_statement_shape_and_chunking():
    session = _RecordingSession()
    repo = VisibilityDataRepository(session)

    n_rows = BULK_UPSERT_CHUNK * 2 + 500
    stored = asyncio.run(repo.bulk_upsert([_row(i) for i in range(n_rows)]))

    assert stored == n_rows
    assert len(session.calls) == 3
    assert len(session.calls[0][1]) == BULK_UPSERT_CHUNK
    assert len(session.calls[1][1]) == BULK_UPSERT_CHUNK
    assert len(session.calls[2][1]) == 500

    sql = str(session.calls[0][0].compile(dialect=postgresql.dialect()))
    assert 'ON CONFLICT ON CONSTRAINT uq_visibility_observation_night DO UPDATE SET' in sql
    assert 'RETURNING' not in sql


def test_bulk_upsert_empty_rows_issues_no_queries():
    session = _RecordingSession()
    repo = VisibilityDataRepository(session)

    assert asyncio.run(repo.bulk_upsert([])) == 0
    assert session.calls == []
