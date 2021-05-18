import astropy.units as u
from astropy.units.quantity import Quantity
import numpy as np
import re
from greedy_max.band import Band
from greedy_max.site import Site
from typing import Dict, List
 
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
    decoder = {'A':'0','B':'1','Q':'0',
                'C':'1','LP':'2','FT':'3',
                'SV':'8','DD':'9'}
    #pattern = '|'.join(map(re.escape, decoder.keys()))
    pattern = '|'.join(decoder.keys())
    def __init__(self, 
                 time_slot_length: Quantity, 
                 weights: List[int] , 
                 total_amount: int, 
                 fpu: Dict[Site, List[str]], 
                 fpur: Dict[Site, List[str]], 
                 grat: Dict[Site, List[str]], 
                 instruments: Dict[Site, List[str]], 
                 lgs: Dict[Site,bool] , 
                 mode: Dict[Site,str], 
                 fpu2b: Dict[str,str], 
                 ifus: Dict[str,str]):
        self.slot_length = time_slot_length
        self.weights = weights
        self.total = total_amount
        self.fpu = fpu
        self.fpur = fpur
        self.grating = grat 
        self.instruments = instruments
        self.laser_guide = lgs
        self.mode = mode
        self.fpu_to_barcode = fpu2b
        self.ifu = ifus

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

    def _decode_mask(self, mask_name: str ) -> str:
        return '1'+ re.sub(f'({TimeSlots.pattern})', 
                           lambda m: TimeSlots.decoder[m.group()], mask_name).replace('-','')[6:]
    
    def is_instrument_available(self, site: Site, instrument: str) -> bool:
        return instrument in self.instruments[site]
    
    def is_mask_available(self, site: str, fpu_mask: str, mask_type: str) -> bool:
        
        barcode = None
        if fpu_mask in self.fpu_to_barcode:
            barcode = self.fpu_to_barcode[fpu_mask]
        else:
            barcode = self._decode_mask(fpu_mask)
        if mask_type == 'FPU':
            return barcode in self.fpu[site] 
        elif mask_type == 'FPUr':
            return barcode in self.fpur[site] 
        elif mask_type == 'GRAT':
            return barcode in self.grating[site]
        else:
            return False
        

