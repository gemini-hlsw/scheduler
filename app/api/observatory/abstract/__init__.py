from abc import ABC
from datetime import timedelta
from typing import FrozenSet, NoReturn, Optional

from astropy.time import Time

# TODO: No type information here because of circular dependencies between common.minimodel and this class.
# TODO: Introduces circular dependencies
# from common.minimodel import Resource, ObservationMode

# TODO: module common has no attribute minimodel
# import common.minimodel


class ObservatoryProperties(ABC):
    """
    Observatory-specific methods that are not tied to other components or
    structures, and allow computations to be implemented in one place.
    """
    _properties: Optional['ObservatoryProperties'] = None

    @staticmethod
    def set_properties(cls) -> NoReturn:
        if not issubclass(cls, ObservatoryProperties):
            raise ValueError('Illegal properties value.')
        ObservatoryProperties._properties = cls()

    @staticmethod
    def _check_properties() -> NoReturn:
        if ObservatoryProperties._properties is None:
            raise ValueError('Properties have not been set.')

    @staticmethod
    def determine_standard_time(resources: FrozenSet,
                                wavelengths: FrozenSet[float],
                                modes: FrozenSet,
                                cal_length: int) -> Time:
        """
        Given the information, determine the length in hours required for calibration
        on a standard star.
        TODO: Is this the correct description?
        TODO: Do we need modes here or can these just be determined from resources?
        TODO: Based on the Gemini code, it seems like the latter is the case, but we do have
        TODO: the obsmode code in the atom extraction which provides an ObservationMode.
        """
        ObservatoryProperties._check_properties()
        return ObservatoryProperties._properties.determine_standard_time(
            resources,
            wavelengths,
            modes,
            cal_length
        )

    @staticmethod
    def is_instrument(resource) -> bool:
        """
        Determine if the given resource is an instrument or not.
        """
        ObservatoryProperties._check_properties()
        return ObservatoryProperties._properties.is_instrument(resource)

    @staticmethod
    def acquisition_time(resource, observation_mode) -> Optional[timedelta]:
        """
        Given a resource, check if it is an instrument, and if so, lookup the
        acquisition time for the specified mode.
        """
        ObservatoryProperties._check_properties()
        return ObservatoryProperties._properties.acquisition_time(resource, observation_mode)
