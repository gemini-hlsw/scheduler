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
    
    async def bulk_create(
        self,
        targets: list[dict],
    ) -> Sequence[Target]:
        """Create multiple targets at once."""
        return await self.create_many(targets)