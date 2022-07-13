import asyncio
import backoff
from gql.transport.websockets import WebsocketsTransport
from gql import Client, gql
from concurrent.futures import ThreadPoolExecutor, as_completed


from common.queries import observation_update, program_update, target_update


class Session:
    
    def __init__(self, url: str = 'http://localhost:8080'):
        self.url = url
    
    async def _query(self, query: gql):
        return self.client.execute(query)

    async def _subscribe(self, session: Client, sub: gql):
        """
        Async generator for gql.Client.subscribe
        """
        async for res in session.subscribe(sub):
            return res
    
    async def subscribe_all(self):
        subs = [observation_update, program_update, target_update]
        tasks = [asyncio.create_task(self.on_update(sub)) for sub in subs]
        return await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    @backoff.on_exception(backoff.expo, Exception, max_time=300)
    async def on_update(self, query: gql):
        client = Client(transport=WebsocketsTransport(url=f'wss://{self.url}/ws'))
        async with client as session:
            return await self._subscribe(session, query)
