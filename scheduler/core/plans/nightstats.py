# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause


from lucupy.minimodel import Band, ProgramID

from dataclasses import dataclass
from typing import final, Mapping

__all__ = [
    'NightStats',
]


@final
@dataclass(frozen=True)
class NightStats:
    time_loss: str
    plan_score: float
    n_toos: int
    completion_fraction: Mapping[Band, int]
    program_completion: Mapping[ProgramID, str]
