from typing import final

from gpp_client import GPPClient

from scheduler.core.meta import Singleton

__all__ = ["gpp", "GPPClientInstance"]


@final
class GPPClientInstance(metaclass=Singleton):
    """Process-wide, lazily-initialized GPPClient.

    The underlying `GPPClient` is instantiated on first access of `.client`
    (so credential resolution does not run at import time) and lives for the
    lifetime of the process.
    """

    _client: GPPClient | None = None

    @property
    def client(self) -> GPPClient:
        if self._client is None:
            self._client = GPPClient()
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


gpp = GPPClientInstance()
