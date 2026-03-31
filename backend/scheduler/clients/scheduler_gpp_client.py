from typing import final
from gpp_client import GPPClient, GPPDirector

from scheduler.core.meta import Singleton

__all__ = ["gpp_client_instance"]


@final
class GppClient(metaclass=Singleton):

    _client: GPPClient
    _director: GPPDirector

    def __init__(self):
        self._client = GPPClient()
        self._director = GPPDirector(self._client)

    @property
    def client(self) -> GPPClient:
        return self._client

    @property
    def director(self) -> GPPDirector:
        return self._director


class _LazyGppClient:
    """Defers GppClient instantiation (and credential resolution) until first access."""

    _instance: GppClient | None = None

    def _get(self) -> GppClient:
        if self._instance is None:
            self._instance = GppClient()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


gpp_client_instance: GppClient = _LazyGppClient()  # type: ignore[assignment]
