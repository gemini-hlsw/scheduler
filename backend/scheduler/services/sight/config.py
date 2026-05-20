import os
from functools import lru_cache

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


def _resolve_db_url() -> str:
    """Return the connection URL with the asyncpg driver scheme.

    Heroku Postgres add-ons set DATABASE_URL with the legacy `postgres://`
    scheme; SQLAlchemy 2.0 requires `postgresql://`, and the in-process
    Calculator requires the async driver, so normalise to
    `postgresql+asyncpg://`.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
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
def get_db_settings() -> DatabaseSettings:
    return DatabaseSettings(url=_resolve_db_url())
