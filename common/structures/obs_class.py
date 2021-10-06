from enum import Enum, unique


@unique
class ObservationClass(Enum):
    Science = 'science'
    ProgramCalibration = 'prog_cal'
    PartnerCalibration = 'partner_cal'