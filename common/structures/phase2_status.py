from enum import Enum


class Phase2Status(Enum):
    PI_TO_COMPLETE = 1
    NGO_TO_REVIEW = 2
    NGO_IN_REVIEW = 3
    GEMINI_TO_ACTIVATE = 4
    ON_HOLD = 5
    INACTIVE = 6
    PHASE_2_COMPLETE = 7
