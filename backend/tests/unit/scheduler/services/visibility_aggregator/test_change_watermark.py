# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
The ODB change watermark is a scheduler_coordination row holding the `since`
timestamp the next aggregator run queries /scheduler/visibility-changes from.
Reading tolerates a missing or corrupt row (caller falls back to a lookback);
writing upserts so the first run creates the row.
"""
import asyncio
from datetime import datetime, timezone

from scheduler.services.visibility_aggregator.coordination import (
    CHANGE_WATERMARK_NAME,
    get_change_watermark,
    set_change_watermark,
)


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _RecordingSession:
    """Captures execute() calls and replays a canned SELECT result."""

    def __init__(self, row=None):
        self.calls = []
        self._row = row

    async def execute(self, stmt, params=None):
        self.calls.append((str(stmt), params))
        return _FakeResult(self._row)


def test_get_watermark_missing_row_returns_none():
    session = _RecordingSession(row=None)

    assert asyncio.run(get_change_watermark(session)) is None
    assert session.calls[0][1] == {"name": CHANGE_WATERMARK_NAME}


def test_get_watermark_null_since_returns_none():
    session = _RecordingSession(row=(None,))

    assert asyncio.run(get_change_watermark(session)) is None


def test_get_watermark_parses_iso_timestamp():
    session = _RecordingSession(row=("2026-07-15T09:00:00+00:00",))

    result = asyncio.run(get_change_watermark(session))

    assert result == datetime(2026, 7, 15, 9, 0, tzinfo=timezone.utc)


def test_get_watermark_unparseable_returns_none():
    session = _RecordingSession(row=("not-a-timestamp",))

    assert asyncio.run(get_change_watermark(session)) is None


def test_set_watermark_upserts_iso_detail():
    session = _RecordingSession()
    since = datetime(2026, 7, 15, 9, 0, tzinfo=timezone.utc)

    asyncio.run(set_change_watermark(session, since))

    sql, params = session.calls[0]
    assert "ON CONFLICT (name) DO UPDATE" in sql
    assert params["name"] == CHANGE_WATERMARK_NAME
    assert params["detail"] == '{"since": "2026-07-15T09:00:00+00:00"}'
