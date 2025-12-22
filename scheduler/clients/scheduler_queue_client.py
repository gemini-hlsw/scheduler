import asyncio
import os
from datetime import datetime
from typing import Optional, Dict, Callable

import aio_pika
from aio_pika import Message, DeliveryMode, ExchangeType
from aio_pika.abc import AbstractIncomingMessage
from lucupy.minimodel import Site
from pydantic import BaseModel, field_serializer, field_validator

from scheduler.core.meta import AsyncSingleton
from scheduler.engine.params import SchedulerParametersV2


__all__ = ["SchedulerQueueClient", "SchedulerEvent"]

class SchedulerEvent(BaseModel):
    trigger_event: str
    time: datetime
    site: Site
    parameters: SchedulerParametersV2

    @field_serializer('site')
    def serialize_site(self, site: Site) -> str:
        """Convert Site enum to string"""
        return site.site_name

    @field_validator('site', mode='before')
    @classmethod
    def parse_site(cls, value):
        """Convert string to Site enum"""
        if isinstance(value, str):
            return Site.GN if value == 'Gemini North' else Site.GS
        return value

class SchedulerQueueClient(metaclass=AsyncSingleton):

    # TODO: Move everything to a proper config file.
    queue_name = 'scheduler_queue_test'
    exchange_name = 'scheduler_exchange_test'
    routing_key = 'scheduler_key_test'
    MAX_PRIORITY = 1 # Binary Priority: 0 regular event/ 1 direct event

    def __init__(self):
        self.amqp_url = os.environ.get('CLOUDAMQP_URL')
        if not self.amqp_url:
            raise EnvironmentError(
                "AMQP_URL or CLOUDAMQP_URL environment variable not found. "
                "The broker connection URL must be provided."
            )
        self.connection: Optional[aio_pika.abc.AbstractRobustConnection] = None
        self.channel: Optional[aio_pika.abc.AbstractChannel] = None
        self.exchange: Optional[aio_pika.abc.AbstractExchange] = None
        self.queue: Optional[aio_pika.abc.AbstractQueue] = None
        self._is_ready = asyncio.Event()
        self._is_shutdown = False

    async def connect(self):
        """
        Establishes the connection and creates a channel.
        """
        if self.connection and not self.connection.is_closed:
            return

        try:
            self.connection = await aio_pika.connect_robust(
                self.amqp_url,
                timeout=10
            )

            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=1)

            self._is_ready.set()

        except Exception as e:
            print(f"Failed to connect: {e}")
            self._is_ready.clear()
            raise

    def is_connected(self) -> bool:
        """
        Checks the status of the connection and channel.
        """
        return (
                self.connection is not None
                and not self.connection.is_closed
                and self.channel is not None
                and not self.channel.is_closed
        )

    async def setup(self):

        """
        Declares a queue and exchange.
        """

        await self._is_ready.wait()
        try:
            # Declare exchange
            self.exchange = await self.channel.declare_exchange(
                name=self.exchange_name,
                type=ExchangeType.DIRECT,
                durable=True
            )
            print(f"Exchange '{self.exchange_name}' declared")

            # Declare queue with priority support
            self.queue = await self.channel.declare_queue(
                name=self.queue_name,
                durable=True,
                arguments={'x-max-priority': self.MAX_PRIORITY}
            )
            print(f"Queue '{self.queue_name}' declared with max priority {self.MAX_PRIORITY}")

            # Bind queue to exchange
            await self.queue.bind(
                exchange=self.exchange,
                routing_key=self.routing_key
            )
            print(f"Queue bound to exchange with routing key '{self.routing_key}'")

        except Exception as e:
            print(f"Failed to setup queue/exchange: {e}")
            raise

    async def add_schedule_event(
        self,
        event: SchedulerEvent,
        reason: str,
        priority: int = 0):
        """
        Publishes a scheduler event to the queue.

        Args:
            event: The scheduler event to publish
            priority: Priority level (0 = regular, 1 = on_demand)
        """

        await self._is_ready.wait()

        if not self.is_connected():
            raise ConnectionError("Not connected to AMQP broker")

        if priority > self.MAX_PRIORITY:
            print(f"Priority {priority} exceeds MAX_PRIORITY {self.MAX_PRIORITY}, capping")
            priority = self.MAX_PRIORITY

        try:
            message = Message(
                body=event.model_dump_json().encode('utf-8'),
                delivery_mode=DeliveryMode.PERSISTENT,
                priority=priority
            )

            await self.exchange.publish(
                message=message,
                routing_key=self.routing_key
            )

            print(f"Published event: trigger='{event.trigger_event}', site={event.site}, priority={priority}")

        except Exception as e:
            print(f"Failed to publish message: {e}")
            raise

    async def consume_events(self, callback: Callable[[SchedulerEvent], None]):
        """
        Starts consuming events from the queue.

        Args:
            callback: Async function to process each event
        """
        await self._is_ready.wait()

        if not self.is_connected():
            raise ConnectionError("Not connected to AMQP broker")

        async def process_message(message: AbstractIncomingMessage):
            async with message.process():
                try:
                    # Parse the event
                    event = SchedulerEvent.model_validate_json(message.body.decode('utf-8'))
                    print(f"Received event: trigger='{event.trigger_event}', site={event.site}")

                    # Call the user's callback
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event)
                    else:
                        callback(event)

                except Exception as e:
                    print(f"Error processing message: {e}")
                    # Message will be rejected and requeued on exception
                    raise

        try:
            print(f"Starting to consume from queue '{self.queue_name}'")
            await self.queue.consume(process_message)

        except Exception as e:
            print(f"Error setting up consumer: {e}")
            raise

    async def close(self):
        """
        Closes the connection gracefully.
        """
        self._is_shutdown = True

        if self.connection and not self.connection.is_closed:
            print("Closing AMQP connection...")
            await self.connection.close()
            print("AMQP connection closed")

        self.connection = None
        self.channel = None
        self.exchange = None
        self.queue = None
        self._is_ready.clear()

async def main():
    """Small snippet for further implementation."""

    async def process_event(event: SchedulerEvent):
        print(f"Processing event: {event.trigger_event} at {event.time} of event: {event.site}")
        await asyncio.sleep(0.1)
    schedule_queue = None

    try:
        schedule_queue = await SchedulerQueueClient.instance()
        await schedule_queue.connect()
        await schedule_queue.setup()

        # Publish a test event
        event_test = SchedulerEvent(
            trigger_event="test",
            time=datetime.now(),
            site=Site.GS,
            parameters=SchedulerParametersV2(
                vis_start=datetime.now(),
                vis_end=datetime.now(),
            )
        )

        await schedule_queue.add_schedule_event(event_test, priority=1)

        await schedule_queue.consume_events(process_event)

        await asyncio.Event().wait()

    except KeyboardInterrupt:
        print("Received shutdown signal")
    except Exception as e:
        print(f"Application error: {e}")
    finally:
        if schedule_queue:
            await schedule_queue.close()
        print("Application shutdown complete")


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication shutting down gracefully.")


