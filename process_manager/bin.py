import datetime


class SchedulerTask:
    def __init__(self, start_time, end_time, priority, mode, scheduler) -> None:

        self.start_time = start_time
        self.end_time = end_time
        self.priority = priority
        self.mode = mode
        self.job_id = hash((self.start_time, self.end_time))
        self.timeout = datetime.timedelta(seconds=10)
        self.scheduler = scheduler

    def __str__(self) -> str:
        return f'{self.job_id}'


class SchedulingBin:
    def __init__(self, start, float_after, length, number_threads, bin_size) -> None:
        
        self.start = start
        self.float_after = float_after
        self.length = length
        self.number_threads = number_threads
        self.bin_size = bin_size
        self.priority_queue = []
        self.running_tasks = []


class RealTimeSchedulingBin(SchedulingBin):
    def __init__(self, start, float_after, length, number_threads, bin_size) -> None:
        super().__init__(start, float_after, length, number_threads, bin_size)
        self.priority_queue = []
        self.running_tasks = []
