from datetime import date, datetime
from typing import Sequence

from sqlalchemy import select, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.services.sight.database.models import Target, TargetNightData
from scheduler.services.sight.database.repositories.base import BaseRepository


class TargetNightDataRepository(BaseRepository[TargetNightData]):
    """Repository for TargetNightData (Stage 1 cache) operations."""
    
    model = TargetNightData
    
    async def get_by_target_site_night(
        self,
        target_id: int,
        site_id: int,
        night_date: date,
    ) -> TargetNightData | None:
        """Get cached data for a specific target, site, and night."""
        stmt = select(TargetNightData).where(
            and_(
                TargetNightData.target_id == target_id,
                TargetNightData.site_id == site_id,
                TargetNightData.night_date == night_date,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_for_target_and_site(
        self,
        target_id: int,
        site_id: int,
        start_date: date,
        end_date: date,
    ) -> Sequence[TargetNightData]:
        """Get cached data for a target at a site within date range."""
        stmt = select(TargetNightData).where(
            and_(
                TargetNightData.target_id == target_id,
                TargetNightData.site_id == site_id,
                TargetNightData.night_date >= start_date,
                TargetNightData.night_date <= end_date,
            )
        ).order_by(TargetNightData.night_date)
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_for_targets_on_night(
        self,
        target_ids: list[int],
        site_id: int,
        night_date: date,
    ) -> Sequence[TargetNightData]:
        """Get cached data for multiple targets on a single night."""
        if not target_ids:
            return []
        stmt = select(TargetNightData).where(
            and_(
                TargetNightData.target_id.in_(target_ids),
                TargetNightData.site_id == site_id,
                TargetNightData.night_date == night_date,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_stale_for_target(
        self,
        target: Target,
        site_id: int | None = None,
    ) -> Sequence[TargetNightData]:
        """Get cached data that is stale (computed before target was updated)."""
        conditions = [
            TargetNightData.target_id == target.id,
            TargetNightData.target_updated_at < target.updated_at,
        ]
        if site_id is not None:
            conditions.append(TargetNightData.site_id == site_id)
        
        stmt = select(TargetNightData).where(and_(*conditions))
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def delete_stale_for_target(self, target: Target) -> int:
        """Delete all stale cached data for a target. Returns count deleted."""
        stmt = delete(TargetNightData).where(
            and_(
                TargetNightData.target_id == target.id,
                TargetNightData.target_updated_at < target.updated_at,
            )
        )
        result = await self.session.execute(stmt)
        return result.rowcount
    
    async def upsert(
        self,
        target_id: int,
        site_id: int,
        night_date: date,
        **kwargs,
    ) -> TargetNightData:
        """Insert or update target night data."""
        existing = await self.get_by_target_site_night(
            target_id, site_id, night_date
        )
        
        if existing:
            return await self.update(existing, **kwargs)
        else:
            return await self.create(
                target_id=target_id,
                site_id=site_id,
                night_date=night_date,
                **kwargs,
            )
    
    async def delete_old_data(self, before_date: date) -> int:
        """Delete cached data older than a given date. Returns count deleted."""
        stmt = delete(TargetNightData).where(
            TargetNightData.night_date < before_date
        )
        result = await self.session.execute(stmt)
        return result.rowcount
