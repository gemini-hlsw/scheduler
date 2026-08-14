# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""One GPP client per event loop.

A `GPPClient` holds sockets bound to the loop that created them, and this process
runs more than one loop: GraphQL is served on the main loop while a scheduler run
executes on its own loop in a worker thread. Sharing one client across both means
sharing sockets across loops, which fails intermittently rather than cleanly.
"""
import asyncio
import threading

import pytest

from scheduler.clients import gpp as gpp_module


class _FakeGPPClient:
    """Stands in for GPPClient so no credentials or sockets are involved."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_clients(monkeypatch):
    monkeypatch.setattr(gpp_module, "GPPClient", _FakeGPPClient)
    gpp_module.gpp._clients.clear()
    yield
    gpp_module.gpp._clients.clear()


def _in_new_loop(coroutine_function):
    """Run a coroutine on a loop of its own, then close it.

    Deliberately not `asyncio.run`: that leaves the thread with no current loop,
    and other suites have sync fixtures that still call `get_event_loop`.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coroutine_function())
    finally:
        loop.close()


def test_one_client_per_loop():
    async def _grab():
        return gpp_module.gpp.client, gpp_module.gpp.client

    first, second = _in_new_loop(_grab)
    assert first is second


def test_a_second_loop_gets_its_own_client():
    """A run's worker-thread loop must not inherit the server loop's sockets."""

    async def _grab():
        return gpp_module.gpp.client

    server_loop_client = _in_new_loop(_grab)
    from_worker_thread = {}

    def _worker():
        from_worker_thread["client"] = _in_new_loop(_grab)

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()

    assert from_worker_thread["client"] is not server_loop_client


def test_close_closes_only_the_running_loops_client():
    async def _run():
        client = gpp_module.gpp.client
        await gpp_module.gpp.close()
        return client, gpp_module.gpp.client

    closed, replacement = _in_new_loop(_run)
    assert closed.closed
    assert replacement is not closed


def test_clients_of_finished_loops_are_forgotten():
    """A loop that ends without closing its client must not leave an entry
    behind for the life of the process."""

    async def _grab():
        return gpp_module.gpp.client

    _in_new_loop(_grab)
    assert len(gpp_module.gpp._clients) == 1

    _in_new_loop(_grab)
    assert len(gpp_module.gpp._clients) == 1


def test_client_needs_a_running_loop():
    """There is no loop to bind sockets to outside a coroutine."""
    with pytest.raises(RuntimeError):
        gpp_module.gpp.client
