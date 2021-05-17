from enum import IntEnum, unique


@unique
class Band(IntEnum):
    """
    The band to which an observation is scheduled.
    """
    Band1 = 1
    Band2 = 2
    Band3 = 3
    Band4 = 4