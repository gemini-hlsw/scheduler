# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from abc import abstractmethod, ABC
from dataclasses import dataclass
from datetime import timedelta
from enum import auto, Enum
from typing import final, Optional

from lucupy.minimodel import Program
from lucupy.types import ZeroTime

from scheduler.config import ConfigurationError


__all__ = [
    'TimeBuffer',
    'create_time_buffer',
]


class TimeBufferType(Enum):
    NONE = auto()
    FLAT_MINUTES = auto()
    PERCENTAGE = auto()


class TimeBuffer(ABC):
    def __call__(self, *args, **kwargs):
        if len(args) != 1 or kwargs:
            raise ValueError("TimeBuffer expects exactly one positional argument (Program) with no keyword arguments.")
        p, = args

        if not isinstance(p, Program):
            raise TypeError("TimeBuffer argument must be an instance of Program.")

        return self._calculate_time(p)

    @abstractmethod
    def _calculate_time(self, program: Program) -> timedelta:
        ...


@final
@dataclass(frozen=True)
class NoTimeBuffer(TimeBuffer):
    def _calculate_time(self, program: Program) -> timedelta:
        return ZeroTime


@final
@dataclass(frozen=True)
class PercentageTimeBuffer(TimeBuffer):
    percentage: float

    def _calculate_time(self, p: Program) -> timedelta:
        return p.program_awarded() * (1.0 + self.percentage)


@final
@dataclass(frozen=True)
class FlatTimeBuffer(TimeBuffer):
    time_amount: timedelta

    def _calculate_time(self, _: Program) -> timedelta:
        return self.time_amount


def create_time_buffer(buffer_type_str: str,
                       buffer_amount: Optional[float]) -> TimeBuffer:
    """
    Create a TimeBuffer based on the configuration parameters in config.yaml.

    Note that all validation is done in here to keep it centralized, and ConfigurationErrors are raised
    if there are incompatible values.
    """
    try:
        time_buffer_type = TimeBufferType[buffer_type_str.upper()]
    except KeyError:
        raise ConfigurationError('buffer_type', buffer_type_str)

    match time_buffer_type:
        case TimeBufferType.NONE:
            return NoTimeBuffer()

        case TimeBufferType.PERCENTAGE:
            if buffer_amount is None or not (0 < buffer_amount < 1):
                raise ConfigurationError('buffer_amount', str(buffer_amount))
            return PercentageTimeBuffer(buffer_amount)

        case TimeBufferType.FLAT_MINUTES:
            if buffer_amount is None or buffer_amount <= 0:
                raise ConfigurationError('buffer_amount', str(buffer_amount))
            return FlatTimeBuffer(time_amount=timedelta(minutes=buffer_amount))

        case _:
            # This only happens if we add something to TimeBufferType and do not implement it here.
            raise ConfigurationError('buffer_type', buffer_type_str)
