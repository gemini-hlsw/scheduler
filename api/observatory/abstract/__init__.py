from astropy.time import Time

from abc import ABC
from typing import Set

from common.minimodel import Observation, ObservationMode, Resource, Site


class ObservatoryProperties(ABC):
    """
    Observatory-specific methods that are not tied to other components or
    structures, and allow computations to be implemented in one place.
    """

    @staticmethod
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
        return 0.0

    @staticmethod
    def has_complementary_modes(obs: Observation, site: Site) -> Set[Observation]:
        """
        Determine if the specified observation has one or more complementary observations
        that can be performed instead at the specified site.

        The default assumption is that there are none unless otherwise implemented.
        """
        return set()
