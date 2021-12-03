from datetime import datetime

DEFAULT_TIMEOUT = 10
DEFAULT_PERIOD = 5
DEFAULT_POOL_SIZE = 10


class SchedulerTask:
    def __init__(self,
                 start_time: datetime,
                 end_time: datetime,
                 target: callable,
                 timeout: int = 10) -> None:

        self.start_time = start_time
        self.end_time = end_time
        self.job_id = hash((self.start_time, self.end_time))
        self.timeout = timeout
        self.target = target
        self.process = None
    
    def __repr__(self):
        return f"{self.target} <- {{Job={self.job_id}, timeout={self.timeout}}}"
