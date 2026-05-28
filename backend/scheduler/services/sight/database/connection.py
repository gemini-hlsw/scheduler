
import json
import logging
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


_logger = logging.getLogger(__name__)


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


async def init_db_engine() -> None:
    """Create the async engine and session factory if DATABASE_URL is set.

    No-op (with an info log) when DATABASE_URL is unset, so app boot and
    scripts that don't touch the DB succeed. Any subsequent ``session_scope``
    call will raise the deferred ``RuntimeError`` instead.
    """
    global _engine, _session_factory
    if _engine is not None:
        return
    settings = get_db_settings()
    if settings is None:
        _logger.info(
            "DATABASE_URL not set; Sight engine left uninitialised."
        )
        return
    url_str = str(settings.url)
    connect_args: dict = {}
    # Heroku Postgres hostnames live under *.amazonaws.com and require SSL.
    # asyncpg doesn't honor `?sslmode=…` in the URL, so we pass it explicitly.
    # Local containers (localhost / 127.0.0.1) skip SSL.
    if "amazonaws.com" in url_str:
        connect_args["ssl"] = "require"
    _engine = create_async_engine(
        url_str,
        pool_size=settings.pool_size,
        max_overflow=settings.pool_overflow,
        echo=settings.echo_sql,
        pool_pre_ping=True,
        json_serializer=_jsonb_serializer,
        connect_args=connect_args,
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
    """Yield an AsyncSession; commits on success, rolls back on exception.

    Raises ``RuntimeError`` lazily when DATABASE_URL is unset — boot and
    import paths stay quiet; only callers that actually want a session fail.
    """
    if _session_factory is None:
        await init_db_engine()
    if _session_factory is None:
        raise RuntimeError(
            "DATABASE_URL is not set; sight DB features are unavailable. "
            "Set DATABASE_URL or run with use_local_visibility=True."
        )
    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
