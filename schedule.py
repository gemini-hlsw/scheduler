from enum import IntEnum, unique
import astropy.units as u
from astropy.units.quantity import Quantity
import numpy as np
from greedy_max import Site


@unique
class Band(IntEnum):
    """
    The band to which an observation is scheduled.
    """
    Band1 = 1
    Band2 = 2
    Band3 = 3
    Band4 = 4


class Observation:
    """
    The data that comprises an observation.
    """
    # Counter to assign indices to observations.
    _counter: int = 0

    def __init__(self, 
                 name: str,
                 band: Band,
                 site: Site,
                 instrument: str,
                 disperser: str,
                 priority: str,
                 time: Quantity,
                 total_time: Quantity,
                 completion: Quantity):
        self.name = name
        self.idx = Observation._counter
        self.site = site
        self.band = band
        self.instrument = instrument
        self.disperser = disperser 
        self.priority = priority
        self.time = time
        self.total_time = total_time
        self.completion = completion  

        Observation._counter += 1

    def calibrate(self, dt):
        if self.disperser and self.instrument:
            tcal = 18.0 * u.min if self.disperser != 'mirror' and \
                            self.disperser != 'null' and \
                            self.instrument != 'GMOS' else 0.0 * u.min
            # save to total time or add units
            return int(np.ceil(tcal.to(u.h) / dt.to(u.h))) 
        return 0


# TODO: Implement scheduling groups
# class SchedulingGroup:
#    def __init__():


class TimeSlots:
    def __init__(self, time_slot_length, weights, total_amount):
        self.slot_length = time_slot_length
        self.weights = weights
        self.total = total_amount

    # TODO: This method can be static and uses no data from TimeSlots.
    # TODO: No idea what these variables are. ni vs nint?
    def intervals(self, empty_slots: np.ndarray) -> np.ndarray:
        ni = len(empty_slots)
        cvec = np.zeros(ni, dtype=int)
        nint = 1
        cvec[0] = nint
        for j in range(1, ni):
            if empty_slots[j] != (empty_slots[j - 1] + 1):
                nint = nint + 1
            cvec[j] = nint 

        idx = np.digitize(cvec, bins=np.arange(ni) + 1)

        return idx
