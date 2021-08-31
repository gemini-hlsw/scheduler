from dataclasses import dataclass
from typing import List
@dataclass
class Visibility:
    visibility: List[float] # visibility indices where visibility meets all constraints 
    hours: List[int]
    hour_angle: List[float] #hour angle
    fraction: List[float] #fraction
    altitude: List[float] #altitude
    azimuth: List[float] #azimuth
    parallactic_angle: List[float] #parallactic angle
    airmass: List[float]
    sky_brightness: List[float]
