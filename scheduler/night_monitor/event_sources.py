# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import List
from enum import Enum

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
    def __init__(self, client):
        super().__init__(client, EventSourceType.RESOURCE)

    def subscriptions(self) -> List[callable]:
        return[
            lambda: self._client.subscribe('resource_edit')
        ]

class WeatherEventSource(EventSource):
    def __init__(self, client):
        super().__init__(client, EventSourceType.WEATHER)

    def subscriptions(self) -> List[callable]:
        return [
            lambda: self._client.subscribe('weather_change')
        ]

class ODBEventSource(EventSource):
    def __init__(self, client):
        super().__init__(client, EventSourceType.ODB)

    def subscriptions(self) -> List[callable]:
        return [
            lambda: self._client.subscribe('observation_edit'),
            lambda: self._client.subscribe('observation_change')
        ]
