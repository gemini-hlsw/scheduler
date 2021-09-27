import re
from typing import Dict, List

from common.structures.site import Site


class Resources:
    """Class to interact with the Resource Mock retrieved information"""

    decoder = {'A':  '0', 'B':  '1', 'Q':  '0',
               'C':  '1', 'LP': '2', 'FT': '3',
               'SV': '8', 'DD': '9'}
    pattern = '|'.join(decoder.keys())

    def __init__(self, 
                 fpu: Dict[Site, List[str]], 
                 fpur: Dict[Site, List[str]], 
                 gratings: Dict[Site, List[str]],
                 instruments: Dict[Site, List[str]], 
                 lgs: Dict[Site, bool],
                 mode: Dict[Site, str],
                 fpu2b: Dict[Site, Dict[str, str]],
                 ifus: Dict[str, Dict[str, str]]):
        self.fpu = fpu
        self.fpur = fpur
        self.fpur[Site.GS].extend(['11013104','11013107','11020601','11022001',
                                    '11023313','11023327','11023328','11023332',
                                    '11023341','11023342','10000009','10005373',
                                    '10005372','10005374','10005375','10005376',
                                    '10005390'])
        self.gratings = gratings
        self.instruments = instruments
        self.laser_guide = lgs
        self.mode = mode
        self.fpu_to_barcode = fpu2b
        self.ifu = ifus 

    @staticmethod
    def _decode_mask(mask_name: str) -> str:
        return '1' + re.sub(f'({Resources.pattern})',
                            lambda m: Resources.decoder[m.group()], mask_name).replace('-', '')[6:]
    
    def is_instrument_available(self, site: Site, instrument: str) -> bool:
        return instrument in self.instruments[site]
    
    def is_disperser_available(self, site: Site, disperser: str) -> bool:
        return disperser in self.gratings[site]

    def is_mask_available(self, site: Site, fpu_mask: str) -> bool:

        barcode = None
        if fpu_mask == 'None':
            return True
        if fpu_mask in self.fpu_to_barcode[site]:
            barcode = self.fpu_to_barcode[site][fpu_mask]
        elif '-' in fpu_mask:
            barcode = Resources._decode_mask(fpu_mask)
        
        return barcode and barcode in self.fpur[site]