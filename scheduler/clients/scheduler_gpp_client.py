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


gpp_client_instance = GppClient()
