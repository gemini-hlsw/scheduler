from gql.transport.websockets import WebsocketsTransport
from gql.transport.aiohttp import AIOHTTPTransport
from gql import Client, gql


class Session:

    def __init__(self, url: str):
        self.ws_client = Client(transport=WebsocketsTransport(url=f'wss://{url}/ws'))
        self.aoi_client = Client(transport=AIOHTTPTransport(url=url),
                                 fetch_schema_from_transport=True)

    def execute(self, query: gql):
        return self.aoi_client.execute(query)

    async def subscribe(self, subscription: gql):
        async for res in self.ws_client.subscribe(subscription):
            yield res
