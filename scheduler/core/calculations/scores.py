# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Dict

from lucupy.minimodel import NightIndex
import numpy.typing as npt

# Scores for the timeslots in a specific night.
NightTimeslotScores = npt.NDArray[float]

# Scores across all nights per timeslot.
# Indexed by night index, and then timeslot index.
Scores = Dict[NightIndex, NightTimeslotScores]
