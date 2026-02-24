# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Any, List, Tuple
from enum import Enum

# Temporary connection to weather service for tests purposes
from gql import Client, gql
from gql.transport.aiohttp_websockets import AIOHTTPWebsocketsTransport
from gql.transport.aiohttp import AIOHTTPTransport

__all__ = [
    "EventSourceType",
    "ResourceEventSource",
    "WeatherEventSource",
    "ODBEventSource",
]

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
        return[
            (ResourceEventSource.RESOURCE_EDIT, lambda x: self._client.subscribe(ResourceEventSource.RESOURCE_EDIT), None)
        ]

class WeatherEventSource(EventSource):

    WEATHER_CHANGE = 'weather_change'

    def __init__(self, client):
        super().__init__(client, EventSourceType.WEATHER)
        self.ws_transport = AIOHTTPWebsocketsTransport(
            url="wss://weather-graphql-ec26c2063b75.herokuapp.com/"
        )
        self.transport = AIOHTTPTransport(
            url="https://weather-graphql-ec26c2063b75.herokuapp.com/"
        )

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
            (WeatherEventSource.WEATHER_CHANGE,
             lambda x: x.subscribe(self.subscription),
             self.ws_weather_client)
        ]

class ODBEventSource(EventSource):

    OBSERVATION_EDIT = 'observation_edit'
    VISIT_EXECUTED = 'visit_executed'

    def __init__(self, client):
        super().__init__(client, EventSourceType.ODB)

    def subscriptions(self) -> List[Tuple[str ,callable]]:
        return [
            (ODBEventSource.OBSERVATION_EDIT, lambda x: self._client.subscribe(ODBEventSource.OBSERVATION_EDIT), None),
            (ODBEventSource.VISIT_EXECUTED, lambda x: self._client.subscribe(ODBEventSource.VISIT_EXECUTED), None)
        ]
