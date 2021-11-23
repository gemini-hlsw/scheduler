from datetime import datetime, timedelta
from scheduler import Scheduler
from multiprocessing import Process, Queue
from enum import Enum
from dataclasses import dataclass


class SchedulerTask:
    def __init__(self,
                 start_time: datetime,
                 end_time: datetime,
                 priority: int,
                 is_realtime: bool,
                 scheduler: Scheduler) -> None:

        self.start_time = start_time
        self.end_time = end_time
        self.priority = priority
        self.is_realtime = is_realtime
        self.job_id = hash((self.start_time, self.end_time))
        self.timeout = timedelta(seconds=10)
        self.scheduler = scheduler
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
        self.priority_queue = []
        self.running_tasks = []
        self.process_queue = Queue()
        
    
    @staticmethod
    def _wrapper(func, queue, args):
        ret = func(args)
        queue.put(ret)

    def run_task(self, task: SchedulerTask) -> None:
        process_args = (task.scheduler.new_schedule, self.process_queue, task.job_id)
        process = Process(target=self._wrapper, args=process_args)
        process.start()
        self.running_tasks.append(task)
        task.process = process

    def wait(self):
        results = []
        for _ in self.running_tasks:
            res = self.process_queue.get()
            results.append(res)
        for task in self.running_tasks:
            process = task.process
            process.join(task.timeout.total_seconds())
            if process.is_alive():
                process.terminate()
                print(f'Task {task.job_id} timed out')
            else:

                print(f'Task {task.job_id} finished')

        return results

    def float_bin(self):
        self.start += self.float_after
        # check if task are still on bounds #
        total_time = self.start + self.length

        for task in self.priority_queue:
            if task.end_time > total_time:
                # remove from queue
                pass
        for task in self.running_tasks:
            if task.end_time  > total_time:
                task.process.terminate()


class RealTimeSchedulingBin(SchedulingBin):
    def __init__(self, start, float_after, length) -> None:
        super().__init__(start, float_after, length, 1, 1)
