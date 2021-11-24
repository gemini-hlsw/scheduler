from datetime import datetime, timedelta
from scheduler import Scheduler
from runner import PriorityRunner
from multiprocessing import Process, Queue
from enum import Enum
from dataclasses import dataclass
from heapq import heappop
import logging
import asyncio
import signal

class NoTaskError(Exception):
    pass

class SchedulerTask:
    def __init__(self,
                 start_time: datetime,
                 end_time: datetime,
                 priority: int,
                 is_realtime: bool,
                 target: callable,
                 timeout: int = 10) -> None:

        self.start_time = start_time
        self.end_time = end_time
        self.priority = priority
        self.is_realtime = is_realtime
        self.job_id = hash((self.start_time, self.end_time))
        self.timeout = timeout
        self.target = target
        self.process = None

    def __str__(self) -> str:
        return f'{self.job_id}'
    
    def __ge__(self, other: 'SchedulerTask') -> bool:
        return self.priority >= other.priority
    
    def __gt__(self, other: 'SchedulerTask') -> bool:
        return self.priority > other.priority
    
    def __le__(self, other: 'SchedulerTask') -> bool:
        return self.priority <= other.priority
    
    def __lt__(self, other: 'SchedulerTask') -> bool:
        return self.priority < other.priority
    
    def __repr__(self):
        return f"{self.target} <- {{prio={self.priority}, timeout={self.timeout}}}"


class BinType(Enum):
    REALTIME = 'realtime'
    STANDARD = 'standard'


@dataclass
class BinConfig:
    bin_type: BinType
    start: datetime
    float_after: timedelta
    length: timedelta
    number_threads: int = 2
    bin_size: int = 5


class SchedulingBin:
    def __init__(self,
                 start: datetime,
                 float_after: timedelta,
                 length: timedelta,
                 number_threads: int,
                 bin_size: int) -> None:
        
        self.start = start
        self.float_after = float_after
        self.length = length
        self.number_threads = number_threads
        self.bin_size = bin_size
        prun = PriorityRunner(self.bin_size)
        prun.add_done_callback(self._schedule_pending)
        self.runner = prun
        self.pending_tasks = []
        self.queue = asyncio.Queue()
        self.accepting = True
        self.priority_queue = []
        self.running_tasks = []        

    def float_bin(self):
        self.start += self.float_after
        # check if task are still on bounds #
        total_time = self.start + self.length

        for task in self.priority_queue:
            if task.end_time > total_time:
                # remove from queue
                pass
        for task in self.running_tasks:
            if task.end_time > total_time:
                task.process.terminate()
    
    def _schedule_with_runner(self, task):
        return self.runner.schedule(Process(target=task.target),
                                    task.priority, task.timeout)
    
    def _schedule_pending(self):
        """
        Schedules the highest priority pending task.

        Only for internal use, and meant to be a callback for the runner, which
        ensures there will be an available slot. DO NOT invoke this method
        directly, at the risk of silently losing task.
        """
        if self.pending_tasks:
            task = heappop(self.pending_tasks)
            logging.debug(f"  - Scheduling pending: {task}, {len(self.pending_tasks)} left")
            self._schedule_with_runner(task)

    def _execute_task(self, task: SchedulerTask) -> None:

        return self.runner.schedule(Process(target=task.target),
                                    task.priority, task.timeout)

    def shutdown(self):
        """
        Attempt to "gracefully" terminate all running tasks.
        """
        self.accepting = False
        self.runner.terminate_all()

    async def run(self, period):
        
        done = asyncio.Event()

        def shutdown():
            done.set()
            SchedulingBin.shutdown()
            asyncio.get_event_loop().stop()
        
        asyncio.get_event_loop().add_signal_handler(signal.SIGINT, shutdown)
        
        while not done.is_set():
            if len(self.priority_queue) > 0:
                task = heappop(self.priority_queue)
                logging.info(f"Scheduling a job for {task}")
                self._execute_task(task)
                await asyncio.sleep(period)
            else:
                raise NoTaskError("No task in queue!")

class RealTimeSchedulingBin(SchedulingBin):
    def __init__(self, start, float_after, length) -> None:
        super().__init__(start, float_after, length, 1, 1)
