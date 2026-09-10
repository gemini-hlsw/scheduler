# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
update_fields must overwrite a target's coordinate fields with fresh ODB values
and always bump updated_at (a SQL now() expression, not a Python timestamp) so
Stage-1 staleness detection recomputes the target's night data.
"""
import asyncio
from unittest.mock import AsyncMock

from sqlalchemy.sql.functions import now as sql_now

from scheduler.services.sight.database.models import Target
from scheduler.services.sight.database.repositories.targets import TargetRepository


def _session() -> AsyncMock:
    session = AsyncMock()
    session.flush = AsyncMock()
    session.refresh = AsyncMock()
    return session


def _target() -> Target:
    return Target(
        id=1,
        name="NGC 1234",
        is_sidereal=True,
        base_ra=10.0,
        base_dec=-20.0,
        pm_ra=1.5,
        pm_dec=-0.5,
        epoch=2000.0,
    )


def test_update_fields_overwrites_coordinates_and_bumps_updated_at():
    session = _session()
    repo = TargetRepository(session)
    target = _target()

    result = asyncio.run(repo.update_fields(
        target,
        base_ra=11.5,
        base_dec=-21.5,
        pm_ra=None,
        pm_dec=None,
        epoch=2015.5,
    ))

    assert result is target
    assert target.base_ra == 11.5
    assert target.base_dec == -21.5
    assert target.pm_ra is None
    assert target.pm_dec is None
    assert target.epoch == 2015.5
    assert isinstance(target.updated_at, sql_now)
    session.flush.assert_awaited_once()
    session.refresh.assert_awaited_once_with(target)


def test_update_fields_bumps_updated_at_even_when_values_unchanged():
    session = _session()
    repo = TargetRepository(session)
    target = _target()

    asyncio.run(repo.update_fields(
        target,
        base_ra=target.base_ra,
        base_dec=target.base_dec,
        pm_ra=target.pm_ra,
        pm_dec=target.pm_dec,
        epoch=target.epoch,
    ))

    assert isinstance(target.updated_at, sql_now)
