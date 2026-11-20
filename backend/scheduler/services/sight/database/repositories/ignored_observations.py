from typing import Sequence

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.services.sight.database.models import IgnoredObservation


class IgnoredObservationRepository:
    """Repository for the observations left out of the plans."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self) -> Sequence[IgnoredObservation]:
        """Every ignored observation, oldest first."""
        stmt = select(IgnoredObservation).order_by(IgnoredObservation.ignored_at, IgnoredObservation.observation_id)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def add(self, observation_ids: Sequence[str]) -> None:
        """Ignore these observations; the ones already ignored keep their date."""
        if not observation_ids:
            return
        stmt = pg_insert(IgnoredObservation).on_conflict_do_nothing(index_elements=["observation_id"])
        await self.session.execute(stmt, [{"observation_id": observation_id} for observation_id in observation_ids])

    async def remove(self, observation_ids: Sequence[str]) -> None:
        """Stop ignoring these observations."""
        if not observation_ids:
            return
        stmt = delete(IgnoredObservation).where(IgnoredObservation.observation_id.in_(observation_ids))
        await self.session.execute(stmt)
