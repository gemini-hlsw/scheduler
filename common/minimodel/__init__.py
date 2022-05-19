from .atom import *
from .constraints import *
from .group import *
from .magnitude import *
from .observation import *
from .program import *
from .qastate import *
from .resource import *
from .semester import *
from .site import *
from .target import *
from .timeallocation import *
from .timingwindow import *
from .too import *

# Type alias for night indices.
NightIndex = int

class Plan:
    """
    A 'plan' is a collection of nighly plans
    """
    def __init__(self, night_duration=10):
        self._time_slots_left = night_duration
        self._groups = []
        
    def add_group(self, group):
        self._groups.append(group)
        self._time_slots_left -= 1 # TODO: clearly a missrepresantion between time allocates and plan timeslots, a proper function is in place
    
    def is_full(self):
        return self._time_slots_left == 0
