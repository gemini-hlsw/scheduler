# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
targets.name has no unique constraint. get_by_names keys its result by name, so
a duplicated name collapses to one row; unique=True must fail instead, the way
the per-name get_by_name does.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.exc import MultipleResultsFound

from scheduler.services.sight.database.models import Target
from scheduler.services.sight.database.repositories.targets import TargetRepository


def _repo(*targets: Target) -> TargetRepository:
    result = SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: list(targets)))
    return TargetRepository(SimpleNamespace(execute=AsyncMock(return_value=result)))


def _target(id_: int, name: str) -> Target:
    return Target(id=id_, name=name, is_sidereal=True)


def test_unique_raises_on_a_duplicated_name():
    repo = _repo(_target(1, "NGC 1"), _target(2, "NGC 1"), _target(3, "NGC 2"))

    with pytest.raises(MultipleResultsFound, match="NGC 1"):
        asyncio.run(repo.get_by_names(["NGC 1", "NGC 2"], unique=True))


def test_unique_returns_every_target_when_names_are_distinct():
    a, b = _target(1, "NGC 1"), _target(2, "NGC 2")

    assert asyncio.run(_repo(a, b).get_by_names(["NGC 1", "NGC 2"], unique=True)) == {"NGC 1": a, "NGC 2": b}


def test_default_keeps_one_row_per_name():
    """The other callers rely on this: no error, one target per name."""
    repo = _repo(_target(1, "NGC 1"), _target(2, "NGC 1"))

    assert list(asyncio.run(repo.get_by_names(["NGC 1"]))) == ["NGC 1"]
