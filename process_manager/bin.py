from datetime import datetime, timedelta

from runner import PriorityRunner
from multiprocessing import Process
from enum import Enum
from dataclasses import dataclass
from heapq import heappop, heappush
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
        self.accepting = True
        self.priority_queue = []
        self.running_tasks = []

    def float_bin(self):
        self.start += self.float_after
        # check if task are still on bounds #
        total_time = self.start + self.length

        # check on pending tasks #
        for task in self.pending_tasks:
            if task.end_time > total_time:
                # remove from queue 
                pass
        # check on running tasks #
        for task in self.running_tasks:
            if task.end_time > total_time:
                self.runner.terminate(task.process)
    
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
            self.running_tasks.append(task)

    def execute_task(self, task):
        """
        Attempts scheduling a task. In case it's not possible right now,
        because other higher priority tasks are holding all the available
        slots, the task will be queued for later scheduling.
        """
        if not self._schedule_with_runner(task):
            logging.debug("  - Had to queue the task, because it can't be scheduled")
            heappush(self.pending_tasks, task)


    def shutdown(self):
        """
        Attempt to "gracefully" terminate all running tasks.
        """
        self.accepting = False
        self.runner.terminate_all()
        self.running_tasks = []

class RealTimeSchedulingBin(SchedulingBin):
    def __init__(self, start, float_after, length) -> None:
        super().__init__(start, float_after, length, 1, 1)
