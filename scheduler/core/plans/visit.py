# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from datetime import datetime
from typing import final, Optional

from lucupy.minimodel import ObservationID, Resource, ObservationClass

__all__ = [
    'Visit',
]


@final
@dataclass(order=True)
class Visit:
    start_time: datetime  # Unsure if this or something else
    obs_id: ObservationID
    obs_class: ObservationClass
    atom_start_idx: int
    atom_end_idx: int
    start_time_slot: int
    time_slots: int
    score: float
    peak_score: float
    instrument: Optional[Resource]
    completion: str
