# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass, field
from datetime import datetime
from typing import final, Optional, List

from lucupy.minimodel import Observation

__all__ = [
    'Visit',
]


@final
@dataclass(order=True)
class Visit:
    start_time: datetime  # Unsure if this or something else
    # Observation is excluded from comparison: it has no ordering, so two visits with the same
    # start_time would raise when compared. Nothing sorts visits naturally, every sort passes key=.
    observation: Observation = field(compare=False)
    atom_start_idx: int
    atom_end_idx: int
    start_time_slot: int
    time_slots: int
    score: float
    peak_score: float
    step_start_idx: Optional[int]
    step_count: Optional[int]
    completion: str
    atom_times: List[int]  # List of times for each atom in the visit, in slots
