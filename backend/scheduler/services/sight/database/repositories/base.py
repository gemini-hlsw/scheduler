from typing import Generic, TypeVar, Sequence
from datetime import date

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from scheduler.services.sight.database.models import Base


ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """
    Base repository providing common CRUD operations.
    
    Subclasses must set the `model` class attribute.
    """
    
    model: type[ModelT]
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, id: int) -> ModelT | None:
        """Get a single record by ID."""
        return await self.session.get(self.model, id)
    
    async def get_all(self) -> Sequence[ModelT]:
        """Get all records."""
        result = await self.session.execute(select(self.model))
        return result.scalars().all()
    
    async def get_many(self, ids: list[int]) -> Sequence[ModelT]:
        """Get multiple records by IDs."""
        if not ids:
            return []
        stmt = select(self.model).where(self.model.id.in_(ids))
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def create(self, **kwargs) -> ModelT:
        """Create a new record."""
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance
    
    async def create_many(self, items: list[dict]) -> Sequence[ModelT]:
        """Create multiple records."""
        instances = [self.model(**item) for item in items]
        self.session.add_all(instances)
        await self.session.flush()
        return instances
    
    async def update(self, instance: ModelT, **kwargs) -> ModelT:
        """Update an existing record."""
        for key, value in kwargs.items():
            setattr(instance, key, value)
        await self.session.flush()
        return instance
    
    async def delete(self, instance: ModelT) -> None:
        """Delete a record."""
        await self.session.delete(instance)
        await self.session.flush()
    
    async def delete_by_id(self, id: int) -> bool:
        """Delete a record by ID. Returns True if deleted."""
        stmt = delete(self.model).where(self.model.id == id)
        result = await self.session.execute(stmt)
        return result.rowcount > 0