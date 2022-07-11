import asyncio
import backoff
from gql.transport.websockets import WebsocketsTransport
from gql.transport.aiohttp import AIOHTTPTransport
from gql import Client, gql

from common.queries import observation_update, program_update, target_update


class Subscription:
    def __init__(self, session, query):
        self.session = session
        self.query = query
    
    async def __aiter__(self):
        for item in self.session.subscribe(self.query):
            yield item


class Session:
    
    def __init__(self):
        pass
    
    async def query(self, query: gql):
        return self.client.execute(query)

    async def subscribe(self, session: Client, sub: gql):
        """
        Async generator for gql.Client.subscribe
        """
        async for res in session.subscribe(sub):
            return res
    
    @backoff.on_exception(backoff.expo, Exception, max_time=300)
    async def subscribe_all(self, url: str):
        
        client = Client(transport=WebsocketsTransport(url=f'wss://{url}/ws'))

        async with client as session:
            return await self.subscribe(session, observation_update)
        
        #program_sub = asyncio.create_task(self._execute_sub(program_update))
        #observation_sub = asyncio.create_task(self._execute_sub(observation_update))
        # target_sub = asyncio.create_task(self._execute_sub(target_update))
        # await asyncio.gather(program_sub, observation_sub, target_sub)


async def execute(session: Client, subscription: gql):
    async for res in session.subscribe(subscription):
        print(res)
    

@backoff.on_exception(backoff.expo, Exception, max_time=300)
async def connection(url: str):
    client = Client(transport=WebsocketsTransport(url=f'wss://{url}/ws'))
    
    async with client as session:
        await execute(session, observation_update)
