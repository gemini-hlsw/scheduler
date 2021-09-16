from enum import Enum, unique
from typing import Optional, Dict,List
from astropy.coordinates import SkyCoord

@unique
class TargetTag(Enum):
   Sidereal = 'sidereal'
   MajorBody = 'major-body'
   Comet = 'comet'
   Asteroid = 'asteroid'



class Target: 

    def __init__(self, 
                 name: str,
                 tag: Optional[TargetTag],
                 magnitudes: Dict[str, List],
                 designation: Optional[str],
                 coordinates: SkyCoord):
             
        self.name = name
        self.tag = tag
        self.magnitudes = magnitudes
        self.designation = designation
        self.coordinates = coordinates 