import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_db_url() -> Optional[str]:
    """Return the connection URL with the asyncpg driver scheme, or None.

    Heroku Postgres add-ons set DATABASE_URL with the legacy ``postgres://``
    scheme; SQLAlchemy 2.0 requires ``postgresql://``, and the in-process
    Calculator requires the async driver, so we normalise to
    ``postgresql+asyncpg://``.

    Returns None when DATABASE_URL is unset so that callers (scripts, the
    RT/local-only validation path) can import this module without the env var
    set. ``init_engine`` no-ops on None; ``session_scope`` raises lazily on
    first DB use. Mirrors the same lazy-init pattern used by
    ``services/redis_client/redis_client.py``.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        return None
    if url.startswith("postgres://"):
        return "postgresql+asyncpg://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and not url.startswith("postgresql+"):
        return "postgresql+asyncpg://" + url[len("postgresql://"):]
    return url


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: PostgresDsn
    pool_size: int = Field(default=5, ge=1, le=20)
    pool_overflow: int = Field(default=10, ge=0, le=20)
    echo_sql: bool = Field(default=False)


@lru_cache
def get_db_settings() -> Optional[DatabaseSettings]:
    """Returns DatabaseSettings, or None when DATABASE_URL is unset.

    Callers must handle None — see ``init_engine`` in
    ``services/sight/database/connection.py`` for the canonical pattern.
    """
    url = _resolve_db_url()
    if url is None:
        return None
    return DatabaseSettings(url=url)
