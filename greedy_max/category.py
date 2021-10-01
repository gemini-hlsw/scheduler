from enum import Enum, unique


@unique
class Category(Enum):
    Science = 'science'
    ProgramCalibration = 'prog_cal'
    PartnerCalibration = 'partner_cal'