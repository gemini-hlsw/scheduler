from enum import Enum


class ExecutionStatus(Enum):
    AUTO = 1
    OBSERVED = 2
    ONGOING = 3
    PENDING = 4
