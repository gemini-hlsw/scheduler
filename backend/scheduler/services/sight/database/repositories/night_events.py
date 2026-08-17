
from datetime import date, datetime
from typing import Sequence

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.services.sight.database.models import NightEvent
from scheduler.services.sight.database.repositories.base import BaseRepository


class NightEventRepository(BaseRepository[NightEvent]):
    """Repository for NightEvent operations."""
    
    model = NightEvent
    
    async def get_by_site_and_night(
        self, 
        site_id: int, 
        night_date: date
    ) -> NightEvent | None:
        """Get night event for a specific site and night."""
        stmt = select(NightEvent).where(
            and_(
                NightEvent.site_id == site_id,
                NightEvent.night_date == night_date,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_window_containing(
        self,
        site_id: int,
        moment: datetime,
    ) -> NightEvent | None:
        """The night whose observing window brackets ``moment``, if any.

        Returns None during the day, when no window is in progress.
        """
        stmt = select(NightEvent).where(
            and_(
                NightEvent.site_id == site_id,
                NightEvent.night_start <= moment,
                NightEvent.night_end >= moment,
            )
        ).order_by(NightEvent.night_start.desc()).limit(1)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_next_after(
        self,
        site_id: int,
        moment: datetime,
    ) -> NightEvent | None:
        """The earliest night starting after ``moment``.

        Used to answer "tonight" during the day, when no window is in progress.
        """
        stmt = select(NightEvent).where(
            and_(
                NightEvent.site_id == site_id,
                NightEvent.night_start > moment,
            )
        ).order_by(NightEvent.night_start).limit(1)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_site_and_date_range(
        self,
        site_id: int,
        start_date: date,
        end_date: date,
    ) -> Sequence[NightEvent]:
        """Get night events for a site within a date range."""
        stmt = select(NightEvent).where(
            and_(
                NightEvent.site_id == site_id,
                NightEvent.night_date >= start_date,
                NightEvent.night_date <= end_date,
            )
        ).order_by(NightEvent.night_date)
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_for_multiple_sites(
        self,
        site_ids: list[int],
        night_date: date,
    ) -> Sequence[NightEvent]:
        """Get night events for multiple sites on a single night."""
        stmt = select(NightEvent).where(
            and_(
                NightEvent.site_id.in_(site_ids),
                NightEvent.night_date == night_date,
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_existing_dates(
        self,
        site_id: int,
        start_date: date,
        end_date: date,
    ) -> set[date]:
        """Get the night dates in the range that already have night events."""
        stmt = select(NightEvent.night_date).where(
            and_(
                NightEvent.site_id == site_id,
                NightEvent.night_date >= start_date,
                NightEvent.night_date <= end_date,
            )
        )
        result = await self.session.execute(stmt)
        return set(result.scalars().all())

    async def exists(self, site_id: int, night_date: date) -> bool:
        """Check if night event exists for site and night."""
        stmt = select(NightEvent.id).where(
            and_(
                NightEvent.site_id == site_id,
                NightEvent.night_date == night_date,
            )
        ).limit(1)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none() is not None
    
    async def upsert(
        self,
        site_id: int,
        night_date: date,
        **kwargs,
    ) -> NightEvent:
        """Insert or update night event."""
        existing = await self.get_by_site_and_night(site_id, night_date)
        
        if existing:
            return await self.update(existing, **kwargs)
        else:
            return await self.create(
                site_id=site_id,
                night_date=night_date,
                **kwargs,
            )
