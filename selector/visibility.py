from dataclasses import dataclass
from typing import List
@dataclass
class Visibility:
    """
    Includes all the information corresponding visibility for an observation.
    Each variable is a series of values, one for each time period of the night.      
    """
    visibility: List[float] # visibility indices where visibility meets all constraints 
    hours: List[float]
    hour_angle: List[float] #hour angle
    fraction: List[float] #fraction
    altitude: List[float] #altitude
    azimuth: List[float] #azimuth
    parallactic_angle: List[float] #parallactic angle
    airmass: List[float]
    sky_brightness: List[float]
