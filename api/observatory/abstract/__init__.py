from abc import abstractmethod, ABC
from datetime import timedelta
from typing import Optional, Set

from astropy import units as u
from astropy.time import Time

from common.minimodel import ObservationMode, Resource


class ObservatoryProperties(ABC):
    """
    Observatory-specific methods that are not tied to other components or
    structures, and allow computations to be implemented in one place.
    """

    @staticmethod
    @abstractmethod
    def determine_standard_time(resources: Set[Resource],
                                wavelengths: Set[float],
                                modes: Set[ObservationMode],
                                cal_length: int) -> Time:
        """
        Given the information, determine the length in hours required for calibration
        on a standard star.
        TODO: Is this the correct description?
        TODO: Do we need modes here or can these just be determined from resources?
        TODO: Based on the Gemini code, it seems like the latter is the case, but we do have
        TODO: the obsmode code in the atom extraction which provides an ObservationMode.
        """
        ...

    @staticmethod
    @abstractmethod
    def is_instrument(resource: Resource) -> bool:
        """
        Determine if the given resource is an instrument or not.
        """
        ...

    @staticmethod
    @abstractmethod
    def acquisition_time(resource: Resource, observation_mode: ObservationMode) -> Optional[timedelta]:
        """
        Given a resource, check if it is an instrument, and if so, lookup the
        acquisition time for the specified mode.
        """
        ...
