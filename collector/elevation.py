from enum import Enum
from typing import Optional


class ElevationType(Enum):
    AIRMASS = 1
    HOUR_ANGLE = 2


def str_to_elevation_type(type_str: Optional[str]) -> Optional[ElevationType]:
    return None if type_str is None or type_str == 'NONE' else ElevationType[type_str]

# NOTE: This should go in the common package for commonly used functions.
def str_to_float(floatstr: Optional[str]) -> Optional[float]:
    return None if floatstr is None or floatstr == 'NULL' else float(floatstr)

class ElevationConstraints:
    MIN_ELEVATION         = -5.0
    MAX_ELEVATION         =  5.0
    MIN_AIRMASS           =  1.0
    MAX_AIRMASS           =  2.3

    def __init__(self,
                 elevation_type: Optional[ElevationType],
                 min_elevation:  Optional[float],
                 max_elevation:  float):
        
        if elevation_type is None:
            elevation_type = ElevationType.AIRMASS
            
        if min_elevation is None or not min_elevation or min_elevation == 0.0:
            min_elevation = ElevationConstraints.MIN_AIRMASS if elevation_type == ElevationType.AIRMASS else ElevationConstraints.MIN_ELEVATION
        
        if max_elevation is None or not max_elevation or max_elevation == 0.0:
            max_elevation = ElevationConstraints.MAX_AIRMASS if elevation_type == ElevationType.AIRMASS else ElevationConstraints.MAX_ELEVATION
            
        self.elevation_type = elevation_type
        self.min_elevation  = min_elevation
        self.max_elevation  = max_elevation


    def __str__(self): 
        return f'{str(self.elevation_type.name)}, {self.min_elevation}, {self.max_elevation}'  
