# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import List, Tuple
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
    RESOURCE_EDIT = 'resource_edit'

    def __init__(self, client):
        super().__init__(client, EventSourceType.RESOURCE)

    def subscriptions(self) -> List[Tuple[str ,callable]]:
        return[
            (ResourceEventSource.RESOURCE_EDIT, lambda: self._client.subscribe(ResourceEventSource.RESOURCE_EDIT))
        ]

class WeatherEventSource(EventSource):

    WEATHER_CHANGE = 'weather_change'

    def __init__(self, client):
        super().__init__(client, EventSourceType.WEATHER)

    def subscriptions(self) -> List[Tuple[str ,callable]]:
        return [
            (WeatherEventSource.WEATHER_CHANGE, lambda: self._client.subscribe(WeatherEventSource.WEATHER_CHANGE))
        ]

class ODBEventSource(EventSource):

    OBSERVATION_EDIT = 'observation_edit'
    VISIT_EXECUTED = 'visit_executed'

    def __init__(self, client):
        super().__init__(client, EventSourceType.ODB)

    def subscriptions(self) -> List[Tuple[str ,callable]]:
        return [
            (ODBEventSource.OBSERVATION_EDIT, lambda: self._client.subscribe(ODBEventSource.OBSERVATION_EDIT)),
            (ODBEventSource.VISIT_EXECUTED, lambda: self._client.subscribe(ODBEventSource.VISIT_EXECUTED))
        ]
