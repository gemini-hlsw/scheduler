import asyncio
from typing import final
from weakref import ReferenceType, ref

from gpp_client import GPPClient

from scheduler.core.meta import Singleton
from scheduler.services import logger_factory

__all__ = ["gpp", "GPPClientInstance"]

_logger = logger_factory.create_logger(__name__, with_id=False)


@final
class GPPClientInstance(metaclass=Singleton):
    """Lazily-initialized `GPPClient`, one per event loop.
    This is implemented so an error or closure in another loop doesn't affect any
    calls the gpp-client does.git
    """

    def __init__(self) -> None:
        self._clients: dict[int, tuple[ReferenceType, GPPClient]] = {}

    @property
    def client(self) -> GPPClient:
        """The client belonging to the running loop.

        Only usable inside a coroutine: with no running loop there is nothing to
        bind the sockets to, and `asyncio.get_running_loop` raises.
        """
        loop = asyncio.get_running_loop()
        key = id(loop)
        entry = self._clients.get(key)
        if entry is not None:
            loop_ref, client = entry
            if loop_ref() is loop:
                return client

        # Only on the rare create path: a run's loop that never closed its client
        # would otherwise leave its entry behind for the life of the process.
        self._drop_finished_loops()
        client = GPPClient()
        self._clients[key] = (ref(loop), client)
        return client

    def _drop_finished_loops(self) -> None:
        """Forget clients whose loop has gone away.

        Their sockets cannot be closed from another loop, so the best available
        outcome is to drop the reference and let them be collected with the loop.
        A loop that wants a clean close should call `close` before it ends.
        """
        dead = [
            key
            for key, (loop_ref, _) in self._clients.items()
            if (referent := loop_ref()) is None or referent.is_closed()
        ]
        for key in dead:
            del self._clients[key]
        if dead:
            _logger.debug(
                f"Dropped {len(dead)} GPP client(s) belonging to finished event "
                f"loops."
            )

    async def close(self) -> None:
        """Close the running loop's client, if it has one.
        """
        loop = asyncio.get_running_loop()
        key = id(loop)
        entry = self._clients.get(key)
        if entry is not None and entry[0]() is loop:
            del self._clients[key]
            await entry[1].close()
        self._drop_finished_loops()


gpp = GPPClientInstance()
