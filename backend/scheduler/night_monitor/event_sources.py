# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Any, List, Tuple
from enum import Enum

# Temporary connection to weather service for tests purposes
from gql import Client, gql
from gql.transport.aiohttp_websockets import AIOHTTPWebsocketsTransport
from gql.transport.aiohttp import AIOHTTPTransport
from urllib.parse import urlparse
from os import environ

__all__ = [
    "EventSourceType",
    "ODB_OPEN_TIMEOUT",
    "ResourceEventSource",
    "WeatherEventSource",
    "ODBEventSource",
]

# Seconds to allow for the ODB websocket handshake. `websockets` defaults to 10, which is
# too tight for an ODB that is recovering: see ODBEventSource._subscribe_to_calculation_updates.
ODB_OPEN_TIMEOUT = float(environ.get('ODB_WS_OPEN_TIMEOUT', 60.0))

class EventSourceType(Enum):
    RESOURCE = 'resource'
    WEATHER = 'weather'
    ODB = 'odb'


class EventSource:
    def __init__(self, client, source_type: EventSourceType):
        self._client = client # TODO: Singleton of gpp-client/ocs-client
        self.source_type = source_type

class ResourceEventSource(EventSource):
    RESOURCE_EDIT = 'resource_edit'

    def __init__(self, client):
        super().__init__(client, EventSourceType.RESOURCE)

    def get_current_state(self) -> Any:
        return

    def subscriptions(self) -> List[Tuple[str ,callable]]:
        # No subscriptions yet!
        return [
            # (
            #     ResourceEventSource.RESOURCE_EDIT,
            #     lambda x: self._client.subscribe(ResourceEventSource.RESOURCE_EDIT),
            #     None
            # )
        ]

class WeatherEventSource(EventSource):

    WEATHER_CHANGE = 'weather_change'

    def __init__(self, client):
        super().__init__(client, EventSourceType.WEATHER)
        weather_url = environ.get('WEATHER_URL', "http://localhost:4000")
        url_parsed = urlparse(weather_url)
        ws_protocol = "wss" if url_parsed.scheme == "https" else "ws"
        weather_ws_url = f"{ws_protocol}://{url_parsed.netloc}{url_parsed.path}"
        self.ws_transport = AIOHTTPWebsocketsTransport(url=weather_ws_url)
        self.transport = AIOHTTPTransport(url=weather_url)

        self.subscription = gql(
            """
            subscription weatherUpdates {
                weatherUpdates {
                    site
                    imageQuality
                    cloudCover
                    windDirection
                    windSpeed
                }
            }
            """
        )

        self.query = gql(
            """
            query Weather {
                weather {
                    site
                    imageQuality
                    cloudCover
                    windDirection
                    windSpeed
                }
            }
        """
        )

        self.ws_weather_client = Client(transport=self.ws_transport)
        self.weather_client = Client(transport=self.transport)


    async def get_current_state(self) -> Any:
        result = await self.weather_client.execute_async(self.query)
        return result['weather']

    def subscriptions(self) -> List[Tuple[str, callable, Any]]:
        return [
            (
                WeatherEventSource.WEATHER_CHANGE,
                lambda x: x.subscribe(self.subscription),
                self.ws_weather_client
            )
        ]

class ODBEventSource(EventSource):

    OBSERVATION_EDIT = 'observation_edit'
    VISIT_EXECUTED = 'visit_executed'

    def __init__(self, client):
        super().__init__(client, EventSourceType.ODB)

    def _subscribe_to_calculation_updates(self):
        """
        Open the obscalc subscription with a handshake timeout we choose.

        `websockets` defaults `open_timeout` to 10s, and an ODB that is briefly recovering
        (a dyno recycle drops connections abruptly and is then unreachable for a few
        seconds) needs longer than that. Observed: a connection died at 26s, the immediate
        reconnect failed at exactly 10.003s, and the next one succeeded and stayed up --
        so the 10s ceiling cost a whole extra reconnect, and every reconnect loses the
        events emitted while the socket is down.

        Reaching through `_graphql` is deliberate and unfortunate: the domain wrapper
        `scheduler.subscribe_to_calculation_updates()` takes no arguments, so there is no
        supported way to pass connection options. The generated method does forward
        `**kwargs` to `execute_ws` and on to `websockets.connect`. Drop this indirection
        once gpp-client exposes the options itself.
        """
        return self._client._graphql.scheduler_observations_updates(
            executable_only=True,
            open_timeout=ODB_OPEN_TIMEOUT,
        )

    def subscriptions(self) -> List[Tuple[str ,callable]]:
        return [
            # (
            #     ODBEventSource.OBSERVATION_EDIT,
            #     lambda x: self._client.subscribe(ODBEventSource.OBSERVATION_EDIT),
            #     None
            # ),
            (
                ODBEventSource.OBSERVATION_EDIT,
                lambda x: self._subscribe_to_calculation_updates(),
                None
            )
        ]
