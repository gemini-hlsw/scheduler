from datetime import datetime
from enum import Enum

DEFAULT_TIMEOUT = 10

class TaskType(Enum):
    """
    Enum for task types
    """
    STANDARD = 'standard'
    REALTIME = 'realtime'


class SchedulerTask:
    """
    A scheduler task that describes scheduling times and the instance of the scheduler to be used (target)

    """
    def __init__(self,
                 start_time: datetime,
                 end_time: datetime,
                 target: callable,
                 timeout: int = DEFAULT_TIMEOUT) -> None:

        self.start_time = start_time
        self.end_time = end_time
        self.job_id = hash((self.start_time, self.end_time))
        self.timeout = timeout
        self.target = target
        self.process = None
    
    def __repr__(self):
        return f"{self.target} <- {{Job={self.job_id}, timeout={self.timeout}}}"
