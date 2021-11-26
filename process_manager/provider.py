import asyncio
from bin import SchedulerTask
import time
from datetime import datetime
from random import randint
from scheduler import Scheduler
import logging
from dataclasses import dataclass


@dataclass
class TaskGenerator:

    def generator(self):
        while True:
            task = SchedulerTask(datetime(2020, 1, 1, 0, 0, 0),
                                 datetime(2020, 1, 1, 0, 0, 0),
                                 randint(0, 10),
                                 False,
                                 Scheduler(randint(3, 15)))
            time.sleep(randint(1, 5))
            yield task
    
    def next(self):
        return next(self.generator())


class TaskProvider:
    """
    Provides tasks to the process manager
    """

    async def producer(self, queue, period):

        generator = TaskGenerator()
        try:
            while True:
                
                print(generator.next())
                result = await asyncio.to_thread(generator.next)
                logging.info(f'Task {result} from producer recieve')
                await queue.put(result)
                logging.info(f'Queue size: {queue.qsize()}')
                await asyncio.sleep(period)
        
        except asyncio.CancelledError:
            ...
    
    async def consumer(self, queue):
        try:
            while True:
                task = await queue.get()
                return task
        except asyncio.CancelledError:
            ...
