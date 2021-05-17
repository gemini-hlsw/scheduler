from enum import Enum, unique


@unique
class Site(Enum):
    """
    The sites (telescopes) available to an observation.
    """
    GS = 'gs'
    GN = 'gn'
