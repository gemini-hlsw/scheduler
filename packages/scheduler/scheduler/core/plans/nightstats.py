# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from lucupy.minimodel import Band, ProgramID

from dataclasses import dataclass
from typing import final, Dict

__all__ = [
    'NightStats',
]

from scheduler.core.types import TimeLossType


@final
@dataclass(frozen=True)
class NightStats:
    time_loss: Dict[TimeLossType, int]
    plan_score: float
    n_toos: int
    completion_fraction: Dict[Band, int]
    program_completion: Dict[ProgramID, str]
