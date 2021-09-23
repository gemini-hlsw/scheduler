from enum import Enum, unique


@unique
class Category(Enum):
    Science = 'science'
    PartnerCalibration = 'partner_cal'     
    ProgramCalibration = 'prog_cal'
