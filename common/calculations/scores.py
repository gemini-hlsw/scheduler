from typing import List

import numpy.typing as npt

# Scores for the timeslots in a specific night.
NightTimeSlotScores = npt.NDArray[float]

# Scores across all nights for the timeslots.
Scores = List[NightTimeSlotScores]