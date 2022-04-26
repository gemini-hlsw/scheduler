from astropy.time import Time
import astropy.units as u

from enum import Enum, EnumMeta
from typing import Set

from api.observatory.abstract import ObservatoryProperties
from common.minimodel import ObservationMode, Resource


class GeminiProperties(ObservatoryProperties):
    """
    Implementation of ObservatoryCalculations specific to Gemini.
    """
    class _InstrumentsMeta(EnumMeta):
        def __contains__(cls, r: Resource) -> bool:
            return any(inst.id in r for inst in cls.__members__.values())

    # Gemini-specific instruments.
    class Instruments(Enum, metaclass=_InstrumentsMeta):
        FLAMINGOS2 = Resource('Flamingos2')
        GNIRS = Resource('GNIRS')
        NIFS = Resource('NIFS')
        IGRINS = Resource('IGRINS')

    # Instruments for which there are set standards.
    _STANDARD_INSTRUMENTS = [Instruments.FLAMINGOS2,
                             Instruments.GNIRS,
                             Instruments.NIFS,
                             Instruments.IGRINS]

    @staticmethod
    def determine_standard_time(resources: Set[Resource],
                                wavelengths: Set[float],
                                modes: Set[ObservationMode],
                                cal_length: int) -> Time:
        """
        Determine the standard star time required for Gemini.
        """
        if cal_length > 1:
            # Check to see if any of the resources are instruments.
            # TODO: We may only want to include specific resources, in which case, modify
            # TODO: Instruments above to be StandardInstruments.
            if any(resource in GeminiProperties._STANDARD_INSTRUMENTS for resource in resources):
                if all(wavelength <= 2.5 for wavelength in wavelengths):
                    return 1.5 * u.h
                else:
                    return 1.0 * u.h
            if ObservationMode.IMAGING in modes:
                return 2.0 * u.h
            return 0.0 * u.h
