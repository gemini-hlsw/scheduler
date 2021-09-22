from typing import Dict, Optional


class Instrument:
    def __init__(self, 
                 name: str, 
                 disperser: str,
                 configuration: Dict[str, Optional[str]]) -> None:
        self.name = name 
        self.disperser = disperser
        self.configuration = configuration
    
    def observation_mode(self) -> str:
        def gmos_mode():         
            if 'MIRROR' in self.disperser:
                return 'imaging'
            # elif searchlist('arcsec', config['fpu']):
            elif any(['arcsec' in fpu for fpu in self.configuration['fpu']]):
                return 'longslit'
            elif any(['IFU' in fpu for fpu in self.configuration['fpu']]):
                return 'ifu'
            elif any(['CUSTOM_MASK' in fpu for fpu in self.configuration['fpu']]):
                return 'mos'

        def flamingos2_mode():
            if any(['LONGSLIT' in fpu for fpu in self.configuration['fpu']]):
                return 'longslit'

            if any(['FPU_NONE' in fpu for fpu in self.configuration['fpu']]) \
                    and any('IMAGING' in fpu for fpu in self.configuration['decker']):
                return 'imaging'
            else:
                'None'

        def niri_mode():
            return 'imaging' if ('NONE' in self.disperser and 'MASK_IMAGING' in self.configuration['mask']) else None

        def gnirs_mode():
            if 'ACQUISITION' in self.configuration['decker'] and 'IN' in self.configuration['acquisitionMirror']:
                return 'imaging'
            elif any('XD' in fpu for fpu in self.configuration['crossDispersed']):
                return 'xd'
            else:
                return 'longslit'

        def gsaoi_mode():
            return 'imaging'

        def nifs_mode():
            return 'ifu'

        instrument_lookup = { 
                            'GMOS-S': gmos_mode,
                            'GMOS-N': gmos_mode,
                            'GSAOI': gsaoi_mode ,
                            'Flamingos2': flamingos2_mode,
                            'NIRI': niri_mode,
                            'NIFS': nifs_mode,
                            'GNIRS': gnirs_mode
                            }   
    
        return instrument_lookup[self.name]() if self.name in instrument_lookup else 'unknown'

    def wavelength(self):
        total_wav = []
        if 'centralWavelength' in self.configuration:
            for wav in self.configuration['centralWavelength']:
                f_wav = float(wav)
                if f_wav not in total_wav:
                    total_wav.append(f_wav)
        elif 'disperserLambda' in self.configuration:
            for wav in self.configuration['disperserLambda']:
                f_wav = float(wav) / 1000
                if f_wav not in total_wav:
                    total_wav.append(f_wav)
        elif 'wavelength' in self.configuration:
            for wav in self.configuration['wavelength']:
                f_wav = float(wav)
                if f_wav not in total_wav:
                    total_wav.append(f_wav)
        return total_wav
