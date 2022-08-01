from typing import Dict, Optional, List


class GMOSConfiguration:
    def __init__(self,
                 fpu: List[str],
                 fpu_widths: List[str],
                 custom_mask: Optional[List[str]]):
        self.fpu = fpu
        self.fpu_widths = fpu_widths
        self.custom_mask = custom_mask


class WavelengthConfiguration:
    def __init__(self,
                 central_wavelength: Optional[List[str]],
                 disperser_lambda: Optional[List[str]],
                 wavelength: Optional[List[str]]) -> None:
        self.central_wavelength = central_wavelength
        self.disperser_lambda = disperser_lambda
        self.wavelength = wavelength


class Instrument:
    def __init__(self, 
                 name: str, 
                 disperser: str,
                 gmos_configuration: Optional[GMOSConfiguration],
                 camera: Optional[str],
                 decker: Optional[str],
                 acquisition_mirror: Optional[str],
                 mask: Optional[str],
                 cross_dispersed: Optional[str],
                 wavelength_config: WavelengthConfiguration) -> None:
        self.name = name
        self.disperser = disperser
        self.gmos_configuration = gmos_configuration
        self.camera = camera  # NIRI only
        self.decker = decker
        self.acquisition_mirror = acquisition_mirror
        self.mask = mask
        self.cross_dispersed = cross_dispersed
        self.wavelength_config = wavelength_config
    
    def observation_mode(self) -> str:
        def gmos_mode():
            if 'MIRROR' in self.disperser:
                return 'imaging'
            # elif searchlist('arcsec', config['fpu']):
            elif any(['arcsec' in fpu for fpu in self.gmos_configuration.fpu]):
                return 'longslit'
            elif any(['IFU' in fpu for fpu in self.gmos_configuration.fpu]):
                return 'ifu'
            elif any(['CUSTOM_MASK' in fpu for fpu in self.gmos_configuration.fpu]):
                return 'mos'

        def flamingos2_mode():

            if self.gmos_configuration is not None:
                if any(['LONGSLIT' in fpu for fpu in self.gmos_configuration.fpu]):
                    return 'longslit'

                if any(['FPU_NONE' in fpu for fpu in self.gmos_configuration.fpu]) \
                        and any('IMAGING' in fpu for fpu in self.decker):
                    return 'imaging'
            else:
                'None'

        def niri_mode():
            return 'imaging' if ('NONE' in self.disperser and 'MASK_IMAGING' in self.mask) else None

        def gnirs_mode():
            if 'ACQUISITION' in self.decker and 'IN' in self.acquisition_mirror:
                return 'imaging'
            elif any('XD' in fpu for fpu in self.cross_dispersed):
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
                            'GSAOI': gsaoi_mode,
                            'Flamingos2': flamingos2_mode,
                            'NIRI': niri_mode,
                            'NIFS': nifs_mode,
                            'GNIRS': gnirs_mode
                            }   
    
        return instrument_lookup[self.name]() if self.name in instrument_lookup else 'unknown'

    def wavelength(self):
        total_wav = []
        
        if self.wavelength_config is not None:
            if self.wavelength_config.central_wavelength is not None:
                for wav in self.wavelength_config.central_wavelength:
                    f_wav = float(wav)
                    if f_wav not in total_wav:
                        total_wav.append(f_wav)
            elif self.wavelength_config.disperser_lambda is not None:
                for wav in self.wavelength_config.disperser_lambda:
                    f_wav = float(wav) / 1000
                    if f_wav not in total_wav:
                        total_wav.append(f_wav)
            elif self.wavelength_config.wavelength is not None:
                for wav in self.wavelength_config.wavelength:
                    f_wav = float(wav)
                    if f_wav not in total_wav:
                        total_wav.append(f_wav)
        return total_wav
