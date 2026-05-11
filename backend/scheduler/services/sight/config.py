from functools import lru_cache

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: PostgresDsn = Field(
        ...,
        description="PostgreSQL connection string",
        examples=["postgresql+asyncpg://user:password@localhost:5432/scheduler"],
    )
    pool_size: int = Field(default=5, ge=1, le=20)
    pool_overflow: int = Field(default=10, ge=0, le=20)
    echo_sql: bool = Field(default=False)


@lru_cache
def get_db_settings() -> DatabaseSettings:
    return DatabaseSettings()
