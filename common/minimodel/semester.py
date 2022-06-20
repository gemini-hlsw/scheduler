from dataclasses import dataclass
from enum import Enum


class SemesterHalf(str, Enum):
    """
    Gemini typically schedules programs for two semesters per year, namely A and B.
    For other observatories, this logic might have to be substantially changed.
    """
    A = 'A'
    B = 'B'


@dataclass(frozen=True, order=True)
class Semester:
    """
    A semester is a period for which programs may be submitted to Gemini and consists of:
    * A four digit year
    * Two semesters during each year, indicated by the SemesterHalf
    """
    year: int
    half: SemesterHalf

    def __str__(self):
        return f'{self.year}{self.half.value}'
