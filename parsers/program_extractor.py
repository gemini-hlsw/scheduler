from abc import ABC, abstractmethod
from typing import List


class ProgramExtractor(ABC):
    @staticmethod
    @abstractmethod
    def extract_program(raw_data: str) -> List[Program]:
        """
        Extracts program level ID:
        * Program ID
        * Name
        * Key
        * Band
        * Rollover Flag
        * Thesis Flag
        * Mode (Queue, Classical)
        * Awarded Time
        * Time Accounting
        * Notes (do we need)?
        * Investigators (do we need)?
        * Scheduling groups (extracted by _extract_group).
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_group(raw_data: str) -> List[SchedulingGroup]:
        """
        Extracts group information, allowing for nested groups:
        * Group name
        * Key
        * Group type will be a subclass of SchedulingGroup, either a SchedulingAndGroup or SchedulingOrGroup.
        * Any observations that are found here will be placed in an explicit SchedulingAndGroup and extracted by
        *    _extract_observation.
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_observation(raw_data: str) -> List[Observation]:
        """
        Extracts observation level data and subdata.
        * Observation ID
        * Title
        * Key
        * Constraints (extracted by _extract_constraints)
        * Priority
        * Phase 2 Status
        * Exec Status Override
        * Priority
        * ToO Override Rapid
        * Interruptible Flag
        * Setup Time Type
        * Acquisition Overhead
        * Target Environment (extracted by _extract_target_environment)
        * Sequence (not sure how to handle this...)
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_constraints(raw_data: str) -> Constraints:
        """
        Extracts observation level constraints.
        * Environmental Conditions
        * Elevation Constraints
        * Target Windows (extracted by _extract_target_window), looped over
        * Laser Clearance Windows (TBD), looped over
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_target_window(raw_data: str) -> TargetWindow:
        """
        Extracts a target window.
        * Target Window Information (TBD)
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_laser_clearance_window(raw_data: str) -> LaserClearanceWindow:
        """
        Extracts a laser clearance window.
        * Laser Clearance Window Information (TBD)
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_target_environment(raw_data: str) -> TargetEnvironment:
        """
        Extracts the target environment for an observation.
        * Primary index
        * Base / asterism (extracted by _extract_target), looped over
        * Guide environment (extracted by _extract_guide_environment)
        * User targets (do we need?) (extracted by _extract_target), looped over
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_guide_environment(raw_data: str) -> GuideEnvironment:
        """
        Extracts the guide environment for the target environment.
        * Guide groups (extracted by _extract_guide_group), looped over
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_guide_group(raw_data: str) -> GuideGroup:
        """
        Extracts a guide group for the guide environment.
        * Name
        * Tag
        * Primary flag
        * Guide probe information (extracted by _extract_guide_probe_information), looped over
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_guide_probe_information(raw_data: str) -> GuideProbeInformation:
        """
        Extracts guide probe information for a guide group.
        * Guide probe key
        * Target (extracted by _extract_target)
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_target(raw_data: str) -> Target:
        """
        Extracts target information.
        * Name
        * Epoch
        * Tag (sidereal, nonsidereal)
        * DeltaRA/Dec
        * RA/Dec
        * Type (guide star, etc)
        * Magnitudes (extracted by _extract_magnitudes)
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_magnitudes(raw_data: str) -> MagnitudeInfo: # Could be List[Magnitude]
        """
        Extracts magnitude information.
        * Magnitude (extracted by _extract_magnitude), looped over
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_magnitude(raw_data: str) -> Magnitude:
        """
        Extracts a magnitude.
        * Name
        * Value
        * System
        """
        NotImplemented

    @staticmethod
    @abstractmethod
    def _extract_sequence(raw_data: str): # Return type unknown for now
        """
        Extracts the sequence.

        This is where we should store sequence information and determine atoms.
        I am not sure how to do this.
        """
        NotImplemented
