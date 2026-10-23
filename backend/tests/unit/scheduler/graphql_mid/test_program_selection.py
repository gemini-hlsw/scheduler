# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
The operation program selection is shared by every browser.

Each change is applied per program and pushed to every subscriber, so two
operators editing the selector see each other's changes instead of the last one
to send overwriting the other.
"""

import pytest
import pytest_asyncio

from scheduler.engine.params import program_selection_store

_SET = """
mutation($ids: [String!]!, $selected: Boolean!) {
  setProgramSelection(programIds: $ids, selected: $selected) { deselected }
}
"""
_RESET = "mutation { resetProgramSelection { deselected } }"
_SUB = "subscription { programSelectionUpdates { deselected } }"


@pytest_asyncio.fixture(autouse=True)
async def clean_selection():
    """The store is a singleton, so leave it as it was found."""
    yield
    await program_selection_store.reset()


async def _set(schema, ids, selected):
    result = await schema.execute(_SET, variable_values={"ids": ids, "selected": selected})
    assert result.errors is None
    return result.data["setProgramSelection"]["deselected"]


@pytest.mark.asyncio
async def test_changes_to_different_programs_both_stick(scheduler_schema):
    await _set(scheduler_schema, ["p-1"], False)

    # A second operator unchecking another program must not re-check p-1.
    assert await _set(scheduler_schema, ["p-2"], False) == ["p-1", "p-2"]
    assert await _set(scheduler_schema, ["p-1"], True) == ["p-2"]


@pytest.mark.asyncio
async def test_reset_checks_everything(scheduler_schema):
    await _set(scheduler_schema, ["p-1", "p-2"], False)

    result = await scheduler_schema.execute(_RESET)

    assert result.errors is None
    assert result.data["resetProgramSelection"]["deselected"] == []


@pytest.mark.asyncio
async def test_subscribers_get_the_current_selection_then_every_change(scheduler_schema):
    await _set(scheduler_schema, ["p-1"], False)

    sub = await scheduler_schema.subscribe(_SUB)
    try:
        # A browser opening now starts from what the others already picked.
        first = await anext(sub)
        assert first.data["programSelectionUpdates"]["deselected"] == ["p-1"]

        await _set(scheduler_schema, ["p-2"], False)
        update = await anext(sub)
        assert update.data["programSelectionUpdates"]["deselected"] == ["p-1", "p-2"]

        await scheduler_schema.execute(_RESET)
        update = await anext(sub)
        assert update.data["programSelectionUpdates"]["deselected"] == []
    finally:
        await sub.aclose()
