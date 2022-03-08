from abc import ABC, abstractmethod
from typing import Set

from common.minimodel import Observation, Site


class ObservatoryCalculations(ABC):
    """
    Observatory-specific methods that are not tied to other components or
    structures, and allow computations to be implemented in one place.
    """

    @staticmethod
    def has_complementary_modes(obs: Observation, site: Site) -> Set[Observation]:
        """
        Determine if the specified observation has one or more complementary observations
        that can be performed instead at the specified site.

        The default assumption is that there are none unless otherwise implemented.
        """
        return set()
