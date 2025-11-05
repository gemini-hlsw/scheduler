from abc import ABC, abstractmethod

__all__ = [
    'EventHandler',
    'ResourceEventHandler',
    'WeatherEventHandler',
    'WeatherEventHandler',
    'ODBEventHandler'
]

class EventHandler(ABC):

    @abstractmethod
    def parse_event(self, raw_event: dict):
        pass
    @abstractmethod
    def handle(self, event):
        pass


class ResourceEventHandler(EventHandler):

    def parse_event(self, raw_event: dict):
        pass
    def handle(self, event):
        pass


class WeatherEventHandler(EventHandler):

    def parse_event(self, raw_event: dict):
        pass

    def handle(self, event):
        pass

class ODBEventHandler(EventHandler):
    def parse_event(self, raw_event: dict):
        pass
    def handle(self, event):
        pass