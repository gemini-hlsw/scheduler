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
    def parse_and_group(data: dict) -> AndGroup:
        ...

    @staticmethod
    @abstractmethod
    def parse_observation(data: dict) -> Program:
        ...

    @staticmethod
    @abstractmethod
    def parse_atom(data: dict, atom_id: int, qa_state: QAState) -> Atom:
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
