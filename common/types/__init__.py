# Basic type aliases for usefulness.
from typing import List, TypeVar, Union
import numpy.typing as npt
from astropy.time import Time

T = TypeVar('T')


ScalarOrNDArray = Union[T, npt.NDArray[T]]
TimeScalarOrNDArray = Union[Time, npt.NDArray[float]]
ListOrNDArray = Union[List[T], npt.NDArray[T]]
