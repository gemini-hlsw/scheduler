import datetime
from scheduler import Scheduler


class SchedulerTask:
    def __init__(self,
                 start_time: datetime.datetime,
                 end_time: datetime.datetime,
                 priority: int,
                 is_realtime: bool,
                 scheduler: Scheduler) -> None:

        self.start_time = start_time
        self.end_time = end_time
        self.priority = priority
        self.is_realtime = is_realtime
        self.job_id = hash((self.start_time, self.end_time))
        self.timeout = datetime.timedelta(seconds=10)
        self.scheduler = scheduler

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


class SchedulingBin:
    def __init__(self,
                 start: datetime.datetime,
                 float_after: datetime.timedelta,
                 length: datetime.timedelta,
                 number_threads: int,
                 bin_size: int) -> None:
        
        self.start = start
        self.float_after = float_after
        self.length = length
        self.number_threads = number_threads
        self.bin_size = bin_size
        self.priority_queue = []
        self.running_tasks = []


class RealTimeSchedulingBin(SchedulingBin):
    def __init__(self, start, float_after, length) -> None:
        super().__init__(start, float_after, length, 1, 1)
        self.priority_queue = []
        self.running_tasks = []
