from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.services.sight.database.models import Site
from scheduler.services.sight.database.repositories.base import BaseRepository


class SiteRepository(BaseRepository[Site]):
    """Repository for Site operations."""
    
    model = Site
    
    async def get_by_name(self, name: str) -> Site | None:
        """Get site by name."""
        stmt = select(Site).where(Site.name == name)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_all_as_dict(self) -> dict[int, Site]:
        """Get all sites as a dict keyed by ID."""
        sites = await self.get_all()
        return {site.id: site for site in sites}
