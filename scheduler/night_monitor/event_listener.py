

class EventListener:

    _sources = {}


    def _consume_subscriptions(self, subscriptions):
        for subscription in subscriptions:
