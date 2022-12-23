# Copyright (c) 2016-2022 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import List

import numpy.typing as npt

# Scores for the timeslots in a specific night.
NightTimeSlotScores = npt.NDArray[float]

# Scores across all nights for the timeslots.
Scores = List[NightTimeSlotScores]
