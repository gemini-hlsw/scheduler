# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from typing import Dict, Tuple, Callable

from pydantic import BaseModel

from scheduler.clients.scheduler_queue_client import SchedulerQueue
from .event_handler import EventHandler, LastPlanMock

__all__ = ['ResourceEventHandler', 'MockResourceEvent', 'PDRQueue']


class MockResourceEvent(BaseModel):
    resource_status: str
    resource_name: str

class PDRQueue:
    """
    Potential Disabled Resource Queue stores events for resources that present a fault,
    but they need time to be fixed and Resource would notify the Scheduler the status.
    If the resource element is not enabled in a certain time, a new element.

    Attributes:
        queue: (Dict[str,)

    """
    def __init__(self, timeout: float = 15.0):
        """
        Initialize the PDR queue where elements are automatically popped after a timeout.

        Args:
            timeout: Default time in seconds before elements are auto-popped and callback executed.
        """

        self.queue = {}
        self.timeout = timeout
        self.tasks = {}
        self.lock = asyncio.Lock()
        self.disabled_resources = set()
        self._counter = 0


    async def _auto_pop(
            self,
            resource_name: str,
            delay: float,
            scheduler_queue: SchedulerQueue,
            callback: callable
    ) -> MockResourceEvent:
        """
        When the timeout gets triggered, the callback function will be called to
        handle the disabled resource event.

        Args:
            resource_name (str): The resource id from Resource.
            delay (float): The delay time in seconds.
            scheduler_queue (SchedulerQueue): Queue to put the scheduler event in.
            callback (callable): The callback function to handle resource event.
        Returns:
            MockResourceEvent: Pop the event from the queue.
        """
        try:
            await asyncio.sleep(delay)

            async with self.lock:
                if resource_name in self.queue:
                    event = self.queue.pop(resource_name)
                    if resource_name in self.tasks:
                        del self.tasks[resource_name]
                    print(f"Item {resource_name} auto-popped: {event}")

                    # Call callback
                    await callback(event, scheduler_queue)

                    return event
        except asyncio.CancelledError:
            pass

    async def put(
            self,
            event: MockResourceEvent,
            scheduler_queue: SchedulerQueue,
            callback: callable,
            timeout: float = 15.0
    ):
        """
        Add element to the queue and start timer to trigger a callback.

        Args:
            event: (MockResourceEvent): Event from Resource that disables a resource.
            scheduler_queue: (SchedulerQueue): Queue to put the scheduler event in.
            callback (callable): The callback function to handle resource event.
            timeout (float): The timeout time in seconds. After this the callback would be call.
        """

        async with self.lock:
            self.queue[event.resource_name] = event
            timer_duration = timeout if timeout is not None else self.timeout
            task = asyncio.create_task(self._auto_pop(event.resource_name, timer_duration, scheduler_queue, callback))
            self.tasks[event.resource_name] = task


    async def force_pop(self, resource_name: str):
        """
        Force the removal of a disabled resource event when an enabled resource event is triggered.

        Args:
            resource_name (str): The resource id from Resource. It has to match the one used in `put`.

        """
        async with self.lock:
            if resource_name in self.queue:
                _ = self.queue.pop(resource_name)
                # Cancel the task
                if resource_name in self.tasks:
                    self.tasks[resource_name].cancel()
                    try:
                        await self.tasks[resource_name]
                    except asyncio.CancelledError:
                        pass
                    del self.tasks[resource_name]

    async def clear(self):
        """Clear all items and cancel all tasks."""
        async with self.lock:
            for task in self.tasks.values():
                task.cancel()

            # Wait for all tasks to be cancelled
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)

            self.tasks.clear()
            self.queue.clear()
            print("Queue cleared")



class ResourceEventHandler(EventHandler):
    """
    Handles all events from Resource.
    Resources can be disabled or enabled and updates are received here.
    """

    def __init__(self, scheduler_queue: SchedulerQueue):
        super().__init__(scheduler_queue)
        self.pdr = PDRQueue()


    def _build_dispatch_map(self) -> Dict[str, Tuple[Callable, Callable]]:
        return {
            "resource_edit": (
                self.parse_resource_edit,
                self._on_resource_edit,
            ),
        }

    @staticmethod
    def parse_resource_edit(raw_event: dict):
        return MockResourceEvent.model_validate(raw_event)

    async def _disabled_callback(self, event: MockResourceEvent):

        await self.scheduler_queue.add_schedule_event(
            reason=f'Resource {event.resource_name} in plan ',
            event=event
        )
        print("Resource is still disabled after timeout")

    async def _on_resource_edit(self, event: MockResourceEvent):
        """
        Handles resource_edit events: any modification that affects a resource.
        This assumes a similar naming structure as in the ODB.
        Args:
             event (MockResourceEvent): A resource edit Event from Resource.
        """

        match event.resource_status:
            case "disabled":
                # If the event is disabling a resource that is on the plan lets put it in the PDR
                last_plan = LastPlanMock()
                if event.resource_name in last_plan.resources():
                    await self.pdr.put(event, self.scheduler_queue, self._disabled_callback)
            case "enabled":
                await self.pdr.force_pop(event.resource_name)
            case _:
                raise RuntimeError(f"Unknown resource status {event.resource_status}")
