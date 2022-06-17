from datetime import datetime
from enum import Enum

class TaskType(Enum):
    """
    Enum for task types
    """
    STANDARD = 'standard'
    REALTIME = 'realtime'


class SchedulerTask:
    """
    A Scheduler task that describes scheduling times and the instance of the base to be used (target)
    """
    def __init__(self,
                 start_time: datetime,
                 target: callable,
                 timeout: int) -> None:

        self.start_time = start_time
        self.job_id = hash((self.start_time, timeout))
        self.timeout = timeout
        self.target = target
        self.process = None
    
    def __repr__(self):
        return f"{self.target} <- {{Job={self.job_id}, timeout={self.timeout}}}"
