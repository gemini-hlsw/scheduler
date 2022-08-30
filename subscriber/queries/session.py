import asyncio
from typing import Awaitable
import backoff
from gql.transport.websockets import WebsocketsTransport
from gql.transport.aiohttp import AIOHTTPTransport
from gql import Client, gql
from .queries import observation_update, program_update, target_update


class Session:
    
    def __init__(self, url: str = 'localhost:8080'):
        self.url = url
    
    async def _query(self, session: Client, query: gql):
        return await session.execute(query)

    async def _subscribe(self, session: Client, sub: gql):
        """
        Async generator for gql.Client.subscribe
        """
        async for res in session.subscribe(sub):
            return res
    
    @backoff.on_exception(backoff.expo, Exception, max_time=300)
    async def subscribe_all(self) -> Awaitable:
        """
        Subscribe to all the subscriptions using one client
        It would return a list of responses containing the status of each task (one per sub)
        and the result of the first task to be completed
        """
        client = Client(transport=WebsocketsTransport(url=f'wss://{self.url}/ws'))
        subs = [observation_update, program_update, target_update]
        async with client as session:
            tasks = [asyncio.create_task(self._subscribe(session, sub)) for sub in subs]
            return await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    @backoff.on_exception(backoff.expo, Exception, max_time=300)
    async def on_update(self, query: gql) -> Awaitable:
        """
        Subscribe to one subscription using one client
        """
        client = Client(transport=WebsocketsTransport(url=f'wss://{self.url}/ws'))
        async with client as session:
            return await self._subscribe(session, query)
    

    async def query(self, query: gql) -> Awaitable:
        """
        Query the server using one client
        """
        # client = Client(transport=WebsocketsTransport(url=f'wss://{self.url}'))
        client = Client(transport=AIOHTTPTransport(url=f'http://{self.url}'))
        async with client as session:
            return await self._query(session, query)
