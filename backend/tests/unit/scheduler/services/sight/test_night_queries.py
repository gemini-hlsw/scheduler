# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
Queries backing the Visibility tab's "tonight" view.

The night+site read must hit ix_visibility_night_site (added in migration 011)
rather than scanning visibility_data, which holds one JSONB-carrying row per
observation per night of the semester. Target names are joined in the same
statement: fetching them per row would be an N+1 over a page of results.

Statement shape is asserted against the compiled SQL — CI has no database.
"""
import asyncio
from datetime import date, datetime, timezone

from sqlalchemy.dialects import postgresql

from scheduler.services.sight.database.repositories.night_events import (
    NightEventRepository,
)
from scheduler.services.sight.database.repositories.visibility_data import (
    VisibilityDataRepository,
)

_NIGHT = date(2026, 7, 29)
_MOMENT = datetime(2026, 7, 29, 6, 0, 0, tzinfo=timezone.utc)


class _CapturingSession:
    """Captures the statement and returns a canned empty result."""

    def __init__(self, rows=()):
        self.statements = []
        self._rows = list(rows)

    async def execute(self, stmt, params=None):
        self.statements.append(stmt)
        return _Result(self._rows)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None

    def scalar_one(self):
        return self._rows[0] if self._rows else 0

    def scalars(self):
        return self

    def first(self):
        return self._rows[0] if self._rows else None


def _sql(stmt) -> str:
    return str(stmt.compile(dialect=postgresql.dialect()))


def test_visible_rows_filter_by_night_and_site_and_join_the_target():
    session = _CapturingSession()
    repo = VisibilityDataRepository(session)

    asyncio.run(repo.get_visible_on_night(
        night_date=_NIGHT, site_id=1, min_remaining_minutes=1, limit=50, offset=100
    ))

    sql = _sql(session.statements[0])
    assert "visibility_data.night_date =" in sql
    assert "visibility_data.site_id =" in sql
    # The index is (night_date, site_id); both must be equality predicates.
    assert "targets" in sql and "JOIN" in sql.upper()
    assert "LIMIT" in sql.upper()
    assert "OFFSET" in sql.upper()


def test_visible_rows_are_ordered_by_remaining_time():
    session = _CapturingSession()
    repo = VisibilityDataRepository(session)

    asyncio.run(repo.get_visible_on_night(night_date=_NIGHT, site_id=1))

    sql = _sql(session.statements[0]).upper()
    assert "ORDER BY" in sql
    assert "REMAINING_MINUTES DESC" in sql


def test_visible_rows_can_exclude_never_visible_observations():
    session = _CapturingSession()
    repo = VisibilityDataRepository(session)

    asyncio.run(repo.get_visible_on_night(
        night_date=_NIGHT, site_id=1, min_remaining_minutes=1
    ))

    assert "remaining_minutes >=" in _sql(session.statements[0])


def test_counting_visible_rows_does_not_join_or_paginate():
    # The total is only a number; joining targets or applying LIMIT would make
    # it both slower and wrong.
    session = _CapturingSession()
    repo = VisibilityDataRepository(session)

    asyncio.run(repo.count_visible_on_night(night_date=_NIGHT, site_id=1))

    sql = _sql(session.statements[0])
    assert "count" in sql.lower()
    assert "LIMIT" not in sql.upper()
    assert "targets" not in sql


def test_night_window_lookup_brackets_the_moment():
    session = _CapturingSession()
    repo = NightEventRepository(session)

    asyncio.run(repo.get_window_containing(site_id=1, moment=_MOMENT))

    sql = _sql(session.statements[0])
    assert "night_events.night_start <=" in sql
    assert "night_events.night_end >=" in sql
    assert "LIMIT" in sql.upper()


def test_next_night_lookup_takes_the_earliest_upcoming_night():
    session = _CapturingSession()
    repo = NightEventRepository(session)

    asyncio.run(repo.get_next_after(site_id=1, moment=_MOMENT))

    sql = _sql(session.statements[0])
    assert "night_events.night_start >" in sql
    assert "ORDER BY night_events.night_start" in sql
    assert "LIMIT" in sql.upper()
