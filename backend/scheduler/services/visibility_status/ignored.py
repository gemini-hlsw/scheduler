# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""The observations the operators left out of the plans.

Kept in memory and saved to ``ignored_observations``. ``visibility_data`` is
never touched: an ignored observation is simply given no visibility when a plan
is built, so undoing it needs nothing recomputed.
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator, Callable, Iterable, Optional

from scheduler.services import logger_factory

_logger = logger_factory.create_logger(__name__)

__all__ = [
    "IgnoredObservation",
    "IgnoredObservationsStore",
    "ignored_observations_store",
]


@dataclass(frozen=True)
class IgnoredObservation:
    """One ignored observation, by reference label (``G-…``)."""

    observation_id: str
    ignored_at: datetime


@asynccontextmanager
async def _repository_scope() -> AsyncIterator:
    """A repository on its own session, committed on exit."""
    from scheduler.services.sight.database.connection import session_scope
    from scheduler.services.sight.database.repositories import IgnoredObservationRepository

    async with session_scope() as session:
        yield IgnoredObservationRepository(session)


class IgnoredObservationsStore:
    """The ignore list: read from memory, written through to the database.

    ``snapshot`` is what a plan build reads. It takes no lock and does no I/O, so
    it is safe from any thread or event loop and costs a build nothing. Changes
    swap in a new frozenset rather than editing the one handed out, and run one
    at a time, so two operators changing the list at once both keep their change.
    """

    def __init__(self, repository_scope: Optional[Callable] = None) -> None:
        self._repository_scope = repository_scope or _repository_scope
        self._rows: tuple[IgnoredObservation, ...] = ()
        self._ids: frozenset[str] = frozenset()
        self._loaded = False
        self._write_lock = asyncio.Lock()

    def snapshot(self) -> frozenset[str]:
        """The ignored reference labels as of now."""
        return self._ids

    async def load(self) -> None:
        """Read the saved list into memory."""
        async with self._write_lock:
            await self._read()

    async def get_all(self) -> list[IgnoredObservation]:
        """The ignored observations, oldest first."""
        if not self._loaded:
            async with self._write_lock:
                if not self._loaded:
                    await self._read()
        return list(self._rows)

    async def ignore(self, observation_ids: Iterable[str]) -> list[IgnoredObservation]:
        """Leave these observations out of the plans from the next build on."""
        labels = _clean(observation_ids)
        async with self._write_lock:
            async with self._repository_scope() as repository:
                await repository.add(labels)
                rows = await repository.get_all()
            self._replace(rows)
        _logger.info(f"Ignoring {labels}; {len(self._ids)} observations ignored in total.")
        return list(self._rows)

    async def restore(self, observation_ids: Iterable[str]) -> list[IgnoredObservation]:
        """Let these observations back into the plans from the next build on."""
        labels = _clean(observation_ids)
        async with self._write_lock:
            async with self._repository_scope() as repository:
                await repository.remove(labels)
                rows = await repository.get_all()
            self._replace(rows)
        _logger.info(f"No longer ignoring {labels}; {len(self._ids)} observations ignored in total.")
        return list(self._rows)

    async def _read(self) -> None:
        async with self._repository_scope() as repository:
            rows = await repository.get_all()
        self._replace(rows)
        _logger.info(f"Ignore list loaded: {len(self._ids)} observations.")

    def _replace(self, rows) -> None:
        self._rows = tuple(IgnoredObservation(row.observation_id, row.ignored_at) for row in rows)
        self._ids = frozenset(row.observation_id for row in self._rows)
        self._loaded = True


def _clean(observation_ids: Iterable[str]) -> list[str]:
    """Trimmed labels, without blanks or repeats, in the order given."""
    return list(dict.fromkeys(label.strip() for label in observation_ids if label and label.strip()))


ignored_observations_store = IgnoredObservationsStore()  # Singleton
