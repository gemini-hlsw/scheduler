# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""The ignore list is read by every plan and written by an operator now and then.

So the read is a plain attribute: no lock, no await, no database. The database
is only touched when the list changes, and it is written before memory, so a
failed write never leaves a plan honouring something that was not saved.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from scheduler.services.sight.database.repositories import IgnoredObservationRepository
from scheduler.services.visibility_status.ignored import IgnoredObservationsStore

_AT = datetime(2026, 11, 12, 3, 0, tzinfo=timezone.utc)
_FIRST = "G-2026B-0001-Q-0001"
_SECOND = "G-2026B-0001-Q-0002"


class _FakeRepository:
    """Stands in for the table."""

    def __init__(self, stored=()):
        self.stored = {label: _AT for label in stored}
        self.fail = False
        self.reads = 0

    async def get_all(self):
        self.reads += 1
        return [SimpleNamespace(observation_id=label, ignored_at=at) for label, at in self.stored.items()]

    async def add(self, observation_ids):
        if self.fail:
            raise RuntimeError("database is down")
        for label in observation_ids:
            self.stored.setdefault(label, _AT)

    async def remove(self, observation_ids):
        if self.fail:
            raise RuntimeError("database is down")
        for label in observation_ids:
            self.stored.pop(label, None)


def _store(repository: _FakeRepository) -> IgnoredObservationsStore:
    @asynccontextmanager
    async def scope():
        yield repository

    return IgnoredObservationsStore(repository_scope=scope)


def test_snapshot_is_empty_and_synchronous_before_any_load():
    store = _store(_FakeRepository([_FIRST]))

    assert store.snapshot() == frozenset()


@pytest.mark.asyncio
async def test_load_brings_back_what_was_saved():
    store = _store(_FakeRepository([_FIRST]))

    await store.load()

    assert store.snapshot() == frozenset({_FIRST})


@pytest.mark.asyncio
async def test_ignoring_saves_and_updates_the_snapshot():
    repository = _FakeRepository()
    store = _store(repository)

    listed = await store.ignore([_FIRST, _SECOND])

    assert set(repository.stored) == {_FIRST, _SECOND}
    assert store.snapshot() == frozenset({_FIRST, _SECOND})
    assert [row.observation_id for row in listed] == [_FIRST, _SECOND]
    assert listed[0].ignored_at == _AT


@pytest.mark.asyncio
async def test_restoring_removes_only_what_was_asked():
    repository = _FakeRepository([_FIRST, _SECOND])
    store = _store(repository)

    listed = await store.restore([_FIRST])

    assert set(repository.stored) == {_SECOND}
    assert store.snapshot() == frozenset({_SECOND})
    assert [row.observation_id for row in listed] == [_SECOND]


@pytest.mark.asyncio
async def test_labels_are_trimmed_and_blank_ones_dropped():
    repository = _FakeRepository()
    store = _store(repository)

    await store.ignore([f"  {_FIRST} ", "", "   "])

    assert set(repository.stored) == {_FIRST}


@pytest.mark.asyncio
async def test_a_failed_save_leaves_the_snapshot_untouched():
    repository = _FakeRepository([_FIRST])
    store = _store(repository)
    await store.load()
    repository.fail = True

    with pytest.raises(RuntimeError):
        await store.ignore([_SECOND])

    assert store.snapshot() == frozenset({_FIRST})


@pytest.mark.asyncio
async def test_listing_reads_the_database_only_once():
    repository = _FakeRepository([_FIRST])
    store = _store(repository)

    await store.get_all()
    await store.get_all()

    assert repository.reads == 1


@pytest.mark.asyncio
async def test_a_snapshot_already_handed_out_does_not_change():
    store = _store(_FakeRepository())
    before = store.snapshot()

    await store.ignore([_FIRST])

    assert before == frozenset()


class _SlowReadRepository(_FakeRepository):
    """Holds the first read until released, as a slow database would."""

    def __init__(self):
        super().__init__()
        self.release = asyncio.Event()
        self.reading = asyncio.Event()

    async def get_all(self):
        rows = await super().get_all()
        if self.reads == 1:
            self.reading.set()
            await self.release.wait()
        return rows


@pytest.mark.asyncio
async def test_two_operators_changing_the_list_at_once_both_stick():
    repository = _SlowReadRepository()
    store = _store(repository)

    first = asyncio.create_task(store.ignore([_FIRST]))
    await repository.reading.wait()
    second = asyncio.create_task(store.ignore([_SECOND]))
    # Give the second change every chance to overtake the first.
    for _ in range(5):
        await asyncio.sleep(0)
    repository.release.set()
    await asyncio.gather(first, second)

    assert store.snapshot() == frozenset({_FIRST, _SECOND})


class _RecordingSession:
    """Captures execute() calls without touching a database."""

    def __init__(self):
        self.statements = []

    async def execute(self, statement, params=None):
        self.statements.append(statement)


@pytest.mark.asyncio
async def test_adding_twice_is_not_an_error():
    session = _RecordingSession()

    await IgnoredObservationRepository(session).add([_FIRST])

    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "ON CONFLICT (observation_id) DO NOTHING" in sql


@pytest.mark.asyncio
async def test_an_empty_change_issues_no_queries():
    session = _RecordingSession()
    repository = IgnoredObservationRepository(session)

    await repository.add([])
    await repository.remove([])

    assert session.statements == []
