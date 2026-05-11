from datetime import date
from typing import Sequence

from sqlalchemy import select, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.services.sight.database.models import VisibilityData


class VisibilityDataRepository:
    """Repository for VisibilityData model."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_observation(
        self,
        observation_id: str,
        start_date: date,
        end_date: date,
    ) -> Sequence[VisibilityData]:
        """Get visibility data for an observation across a date range."""
        stmt = select(VisibilityData).where(
            and_(
                VisibilityData.observation_id == observation_id,
                VisibilityData.night_date >= start_date,
                VisibilityData.night_date <= end_date,
            )
        ).order_by(VisibilityData.night_date)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_target(
        self,
        target_id: int,
        start_date: date,
        end_date: date,
    ) -> Sequence[VisibilityData]:
        """Get visibility data for a target across a date range."""
        stmt = select(VisibilityData).where(
            and_(
                VisibilityData.target_id == target_id,
                VisibilityData.night_date >= start_date,
                VisibilityData.night_date <= end_date,
            )
        ).order_by(VisibilityData.night_date)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_observation_and_target(
        self,
        observation_id: str,
        target_id: int,
        start_date: date,
        end_date: date,
    ) -> Sequence[VisibilityData]:
        """Get visibility data for a specific observation and target."""
        stmt = select(VisibilityData).where(
            and_(
                VisibilityData.observation_id == observation_id,
                VisibilityData.target_id == target_id,
                VisibilityData.night_date >= start_date,
                VisibilityData.night_date <= end_date,
            )
        ).order_by(VisibilityData.night_date)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_observation_ids_on_night(
        self,
        observation_ids: list[str],
        night_date: date,
    ) -> Sequence[VisibilityData]:
        """Get visibility data for multiple observations on a single night."""
        if not observation_ids:
            return []
        stmt = select(VisibilityData).where(
            and_(
                VisibilityData.observation_id.in_(observation_ids),
                VisibilityData.night_date == night_date,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def upsert(
        self,
        observation_id: str,
        target_id: int,
        site_id: int,
        night_date: date,
        remaining_minutes: int,
        visible_ranges: list,
        constraints: dict,
    ) -> VisibilityData:
        """Insert or update visibility data."""
        stmt = select(VisibilityData).where(
            and_(
                VisibilityData.observation_id == observation_id,
                VisibilityData.target_id == target_id,
                VisibilityData.site_id == site_id,
                VisibilityData.night_date == night_date,
            )
        )
        result = await self.session.execute(stmt)
        existing = result.scalar_one_or_none()
        
        if existing:
            existing.remaining_minutes = remaining_minutes
            existing.visible_ranges = visible_ranges
            existing.constraints = constraints
            await self.session.flush()
            return existing
        
        data = VisibilityData(
            observation_id=observation_id,
            target_id=target_id,
            site_id=site_id,
            night_date=night_date,
            remaining_minutes=remaining_minutes,
            visible_ranges=visible_ranges,
            constraints=constraints,
        )
        self.session.add(data)
        await self.session.flush()
        return data

    async def delete_by_observation(
        self,
        observation_id: str,
    ) -> int:
        """Delete all visibility data for an observation."""
        stmt = delete(VisibilityData).where(
            VisibilityData.observation_id == observation_id
        )
        result = await self.session.execute(stmt)
        return result.rowcount
