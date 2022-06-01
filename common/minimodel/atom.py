from dataclasses import dataclass
from datetime import timedelta
from typing import FrozenSet

from .qastate import QAState
from .resource import Resource


@dataclass
class Atom:
    """
    Atom information, where an atom is the smallest schedulable set of steps
    such that useful science can be obtained from performing them.
    Wavelengths must be specified in microns.
    """
    id: int
    exec_time: timedelta
    prog_time: timedelta
    part_time: timedelta
    observed: bool
    qa_state: QAState
    guide_state: bool
    resources: FrozenSet[Resource]
    wavelengths: FrozenSet[float]
