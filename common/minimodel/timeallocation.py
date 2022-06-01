from dataclasses import dataclass
from datetime import timedelta
from enum import Enum


class TimeAccountingCode(str, Enum):
    """
    The time accounting codes for the possible partner submissions or internal program
    types used at Gemini, also known as categories.

    This will have to be customized for a given observatory if used independently
    of Gemini.
    """
    AR = 'Argentina'
    AU = 'Australia'
    BR = 'Brazil'
    CA = 'Canada'
    CFH = 'CFHT Exchange'
    CL = 'Chile'
    KR = 'Republic of Korea'
    DD = "Director's Time"
    DS = 'Demo Science'
    GS = 'Gemini Staff'
    GT = 'Guaranteed Time'
    JP = 'Subaru'
    LP = 'Large Program'
    LTP = 'Limited-term Participant'
    SV = 'System Verification'
    UH = 'University of Hawaii'
    UK = 'United Kingdom'
    US = 'United States'
    XCHK = 'Keck Exchange'


@dataclass
class TimeAllocation:
    """
    Time allocation information for a given category for a program.
    Programs may be sponsored by multiple categories with different amounts
    of time awarded. This class maintains information about the time awarded
    and the time that has been used, divided between program time and partner
    calibration time. The time used is calculated as a ratio of the awarded time
    for this category to the total time awarded to the program.
    """
    category: TimeAccountingCode
    program_awarded: timedelta
    partner_awarded: timedelta
    program_used: timedelta
    partner_used: timedelta

    def total_awarded(self) -> timedelta:
        return self.program_awarded + self.partner_awarded

    def total_used(self) -> timedelta:
        return self.program_used + self.partner_used

    def __hash__(self):
        return self.category.__hash__()
