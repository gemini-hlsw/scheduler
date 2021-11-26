from abc import ABC, abstractmethod
from typing import Mapping
from ..minimodel import *


class ProgramProvider(ABC):
    """
    The base class for all populators of the mini-model.

    In all cases, this information will be provided in JSON format, which will
    be a dict of key - entry pairs.

    The values of the entries in these pairs will be int, float, or str, so we
    leave them as generic dict.
    """
    @staticmethod
    @abstractmethod
    def parse_program(json: dict) -> Program:
        ...

    @staticmethod
    @abstractmethod
    def parse_or_group(json: dict) -> OrGroup:
        ...

    @staticmethod
    @abstractmethod
    def parse_and_group(json: dict) -> AndGroup:
        ...

    @staticmethod
    @abstractmethod
    def parse_observation(json: dict) -> Program:
        ...

    @staticmethod
    @abstractmethod
    def parse_atom(json: dict) -> Observation:
        ...

    @staticmethod
    @abstractmethod
    def parse_sidereal_target(json: dict) -> SiderealTarget:
        ...

    @staticmethod
    @abstractmethod
    def parse_nonsidereal_target(json: dict) -> NonsiderealTarget:
        ...

    @staticmethod
    @abstractmethod
    def parse_magnitude(json: dict) -> Magnitude:
        ...

    @staticmethod
    @abstractmethod
    def parse_constraints(json: dict) -> Constraints:
        ...

    @staticmethod
    @abstractmethod
    def parse_timing_window(json: dict) -> TimingWindow:
        ...

    @staticmethod
    @abstractmethod
    def parse_time_allocation(json: dict) -> TimeAllocation:
        ...
