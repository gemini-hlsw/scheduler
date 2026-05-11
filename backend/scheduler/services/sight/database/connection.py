"""Async SQLAlchemy engine + session lifecycle for the sight service.

The engine is created lazily via ``init_engine()`` (called from the FastAPI
lifespan in ``app.py``) so importing this module does not require ``DB_URL``
to be set. Use ``session_scope()`` to obtain a session for any in-process
caller; it commits on success and rolls back on exception.
"""

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import date, datetime
from decimal import Decimal
from uuid import UUID

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from scheduler.services.sight.config import get_db_settings


def _jsonb_default(obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, UUID):
        return str(obj)
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _jsonb_serializer(value) -> str:
    return json.dumps(value, default=_jsonb_default)


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_engine() -> None:
    """Create the async engine and session factory. Idempotent."""
    global _engine, _session_factory
    if _engine is not None:
        return
    settings = get_db_settings()
    _engine = create_async_engine(
        str(settings.url),
        pool_size=settings.pool_size,
        max_overflow=settings.pool_overflow,
        echo=settings.echo_sql,
        pool_pre_ping=True,
        json_serializer=_jsonb_serializer,
    )
    _session_factory = async_sessionmaker(
        bind=_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


async def dispose_engine() -> None:
    """Dispose the engine and clear the singleton state."""
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _session_factory = None


@asynccontextmanager
async def session_scope() -> AsyncGenerator[AsyncSession, None]:
    """Yield an AsyncSession; commits on success, rolls back on exception."""
    if _session_factory is None:
        await init_engine()
    assert _session_factory is not None
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
