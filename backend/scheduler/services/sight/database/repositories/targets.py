from datetime import datetime
from typing import Sequence

from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.services.sight.database.models import Target
from scheduler.services.sight.database.repositories.base import BaseRepository


class TargetRepository(BaseRepository[Target]):
    """Repository for Target operations."""
    
    model = Target
    
    async def get_by_name(self, name: str) -> Target | None:
        """Get target by name."""
        stmt = select(Target).where(Target.name == name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_names(self, names: list[str]) -> dict[str, Target]:
        """Get all targets whose name is in `names`, keyed by name. One query."""
        if not names:
            return {}
        stmt = select(Target).where(Target.name.in_(set(names)))
        result = await self.session.execute(stmt)
        return {t.name: t for t in result.scalars().all()}
    
    async def get_ids_by_names(self, names: list[str]) -> dict[str, int]:
        """Ids of the targets whose name is in `names`, keyed by name.

        Ids and names only, for callers checking whether a target exists;
        `get_by_names` fetches every column of every matching row.
        """
        if not names:
            return {}
        stmt = select(Target.name, Target.id).where(Target.name.in_(set(names)))
        result = await self.session.execute(stmt)
        return {name: id_ for name, id_ in result.all()}

    async def get_sidereal(self) -> Sequence[Target]:
        """Get all sidereal targets."""
        stmt = select(Target).where(Target.is_sidereal == True)
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_non_sidereal(self) -> Sequence[Target]:
        """Get all non-sidereal targets (require Horizons)."""
        stmt = select(Target).where(Target.is_sidereal == False)
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_updated_since(self, since: datetime) -> Sequence[Target]:
        """Get targets updated since a given time."""
        stmt = select(Target).where(Target.updated_at >= since)
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def touch(self, target: Target) -> Target:
        """Update the updated_at timestamp to trigger recalculation."""
        target.updated_at = func.now()
        await self.session.flush()
        await self.session.refresh(target)
        return target

    async def update_fields(
        self,
        target: Target,
        *,
        base_ra: float | None,
        base_dec: float | None,
        pm_ra: float | None,
        pm_dec: float | None,
        epoch: float | None,
    ) -> Target:
        """Update a sidereal target's coordinate fields from fresh ODB values.

        Bumps updated_at explicitly so Stage-1 rows become stale and get
        recomputed even if every field value is unchanged.
        """
        target.base_ra = base_ra
        target.base_dec = base_dec
        target.pm_ra = pm_ra
        target.pm_dec = pm_dec
        target.epoch = epoch
        target.updated_at = func.now()
        await self.session.flush()
        await self.session.refresh(target)
        return target
    
    async def bulk_create(
        self,
        targets: list[dict],
    ) -> Sequence[Target]:
        """Create multiple targets at once."""
        return await self.create_many(targets)