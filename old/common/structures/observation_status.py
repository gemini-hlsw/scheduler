from enum import Enum


class ObservationStatus(Enum):
    OBSERVED = 1
    ONGOING = 2
    READY = 3
    PHASE2 = 4
    FOR_REVIEW = 5
    IN_REVIEW = 6
    FOR_ACTIVATION = 7
    ON_HOLD = 8
    INACTIVE = 9
