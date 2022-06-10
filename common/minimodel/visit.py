from datetime import datetime
from .observation import ObservationID
from dataclasses import dataclass


@dataclass(order=True, frozen=True)
class Visit:
    start_time: datetime  # Unsure if this or something else
    obs_id: ObservationID
    atom_start_idx: int
    atom_end_idx: int
