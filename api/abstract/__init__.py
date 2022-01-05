from common.minimodel import *


class ProgramProvider(ABC):
    """
    The base class for all populators of the mini-model.

    In all cases, this information will be provided in JSON format, which will
    be a dict of key - entry pairs.

    The values of the entries in these pairs will be int, float, or str, so we
    leave them as generic dict.

    Methods can and should only should raise:
    * ValueError if the data is parseable but not of the correct type
    * TypeError if the data is of the wrong type
    * KeyError if a lookup fails in an enum
    * NotImplementedError if the feature is not offered in this provider
    """
    @staticmethod
    @abstractmethod
    def parse_program(data: dict) -> Program:
        ...

    @staticmethod
    @abstractmethod
    def parse_or_group(data: dict) -> OrGroup:
        ...

    @staticmethod
    @abstractmethod
    def parse_and_group(data: dict, group_id: str, group_name: str) -> AndGroup:
        ...

    @staticmethod
    @abstractmethod
    def parse_observation(data: dict, num: int) -> Program:
        ...

    @staticmethod
    @abstractmethod
    def parse_target(data: dict) -> Target:
        ...

    @staticmethod
    @abstractmethod
    def parse_sidereal_target(data: dict) -> SiderealTarget:
        ...

    @staticmethod
    @abstractmethod
    def parse_nonsidereal_target(data: dict) -> NonsiderealTarget:
        ...

    @staticmethod
    @abstractmethod
    def parse_magnitude(data: dict) -> Magnitude:
        ...

    @staticmethod
    @abstractmethod
    def parse_constraints(data: dict) -> Constraints:
        ...

    @staticmethod
    @abstractmethod
    def parse_timing_window(data: dict) -> TimingWindow:
        ...

    @staticmethod
    @abstractmethod
    def parse_time_allocation(data: dict) -> TimeAllocation:
        ...

    @staticmethod
    @abstractmethod
    def parse_atoms(sequence: List[dict], qa_states: List[QAState]) -> List[Atom]:
        ...
