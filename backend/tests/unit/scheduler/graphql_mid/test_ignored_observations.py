# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""The ignore list over GraphQL, with the store stubbed: the resolver contract only.

The list is shared by every browser, so each change is pushed to all of them.
"""

from datetime import datetime, timezone

import pytest

from scheduler.graphql_mid import schema as schema_module
from scheduler.services.visibility_status.ignored import IgnoredObservation

_AT = datetime(2026, 11, 12, 3, 0, tzinfo=timezone.utc)
_FIRST = "G-2026B-0001-Q-0001"
_SECOND = "G-2026B-0001-Q-0002"


class _FakeStore:
    def __init__(self, labels=()):
        self.labels = list(labels)
        self.calls = []

    def _rows(self):
        return [IgnoredObservation(observation_id=label, ignored_at=_AT) for label in self.labels]

    async def get_all(self):
        return self._rows()

    async def ignore(self, observation_ids):
        self.calls.append(("ignore", list(observation_ids)))
        self.labels += [label for label in observation_ids if label not in self.labels]
        return self._rows()

    async def restore(self, observation_ids):
        self.calls.append(("restore", list(observation_ids)))
        self.labels = [label for label in self.labels if label not in observation_ids]
        return self._rows()


@pytest.fixture
def store(monkeypatch):
    fake = _FakeStore([_FIRST])
    monkeypatch.setattr(schema_module, "ignored_observations_store", fake)
    return fake


@pytest.mark.asyncio
async def test_the_ignore_list_is_readable(scheduler_schema, store):
    result = await scheduler_schema.execute("query { ignoredObservations { observationId ignoredAt } }")

    assert result.errors is None
    assert result.data["ignoredObservations"] == [{"observationId": _FIRST, "ignoredAt": _AT.isoformat()}]


@pytest.mark.asyncio
async def test_ignoring_returns_the_new_list(scheduler_schema, store):
    result = await scheduler_schema.execute(
        "mutation($ids: [String!]!) { setObservationsIgnored(observationIds: $ids, ignored: true) { observationId } }",
        variable_values={"ids": [_SECOND]},
    )

    assert result.errors is None
    assert store.calls == [("ignore", [_SECOND])]
    assert [row["observationId"] for row in result.data["setObservationsIgnored"]] == [_FIRST, _SECOND]


@pytest.mark.asyncio
async def test_undoing_returns_the_new_list(scheduler_schema, store):
    result = await scheduler_schema.execute(
        "mutation($ids: [String!]!) { setObservationsIgnored(observationIds: $ids, ignored: false) { observationId } }",
        variable_values={"ids": [_FIRST]},
    )

    assert result.errors is None
    assert store.calls == [("restore", [_FIRST])]
    assert result.data["setObservationsIgnored"] == []


_SUB = "subscription { ignoredObservationsUpdates { observationId } }"
_SET = "mutation($ids: [String!]!, $ignored: Boolean!) { setObservationsIgnored(observationIds: $ids, ignored: $ignored) { observationId } }"


def _labels(message) -> list[str]:
    return [row["observationId"] for row in message.data["ignoredObservationsUpdates"]]


@pytest.mark.asyncio
async def test_subscribers_get_the_current_list_then_every_change(scheduler_schema, store):
    sub = await scheduler_schema.subscribe(_SUB)
    try:
        # A browser opening now starts from what the others already ignored.
        assert _labels(await anext(sub)) == [_FIRST]

        await scheduler_schema.execute(_SET, variable_values={"ids": [_SECOND], "ignored": True})
        assert _labels(await anext(sub)) == [_FIRST, _SECOND]

        await scheduler_schema.execute(_SET, variable_values={"ids": [_FIRST], "ignored": False})
        assert _labels(await anext(sub)) == [_SECOND]
    finally:
        await sub.aclose()


@pytest.mark.asyncio
async def test_a_closed_subscription_is_forgotten(scheduler_schema, store):
    from scheduler.shared_queue import ignored_observations_subscribers

    sub = await scheduler_schema.subscribe(_SUB)
    await anext(sub)
    assert len(ignored_observations_subscribers) == 1

    await sub.aclose()

    assert len(ignored_observations_subscribers) == 0
