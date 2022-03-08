from typing import Set

from api.observatory.abstract import ObservatoryCalculations
from common.minimodel import Observation, Site


class GeminiCalculations(ObservatoryCalculations):
    """
    Implementation of ObservatoryCalculations specific to Gemini.
    TODO: See and adapt old code below from old Selector.
    """
    @staticmethod
    def has_complementary_modes(obs: Observation, site: Site) -> Set[Observation]:
        """
        go = False
        altinst = 'None'

        obs_site = site
        mode = obs.instrument.observation_mode()
        instrument = obs.instrument
        if obs_site != site:
            if site == Site.GN:
                if (instrument.name == 'GMOS-S' and
                        mode in ['imaging', 'longslit', 'ifu']):
                    go = True
                    altinst = 'GMOS-N'
                elif instrument.name == 'Flamingos2':
                    if mode in ['imaging']:
                        go = True
                        altinst = 'NIRI'
                    elif (mode in ['longslit'] and
                          'GCAL' not in instrument.name and
                          'R3000' in instrument.disperser):
                        go = True
                        altinst = 'GNIRS'
            elif site == Site.GN:

                if (instrument.name == 'GMOS-N' and
                        mode in ['imaging', 'longslit', 'ifu']):
                    go = True
                    altinst = 'GMOS-S'
                elif (instrument.name == 'NIRI' and
                      instrument.camera == 'F6'):
                    go = True
                    altinst = 'Flamingos2'
                    if mode in ['imaging']:
                        go = True
                        altinst = 'Flamingos2'
                    elif mode in ['longslit']:
                        if (instrument.disperser == 'D_10' and
                                'SHORT' in instrument.camera):
                            go = True
                            altinst = 'Flamingos2'
                        elif (instrument.disperser == 'D_10' and
                              'LONG' in instrument.camera):
                            go = True
                            altinst = 'Flamingos2'
        else:
            go = True
            altinst = obs.instrument.name

        return go, altinst
        """
        ...
