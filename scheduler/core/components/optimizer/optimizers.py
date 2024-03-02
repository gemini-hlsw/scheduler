# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from enum import Enum

from .base_optimizer import BaseOptimizer
from .dummy_optimizer import DummyOptimizer
from .greedy_max_optimizer import GreedyMaxOptimizer

from lucupy.types import Instantiable


__all__ = [
    'Optimizers',
]


class Optimizers(Instantiable[BaseOptimizer], Enum):
    DUMMY = Instantiable(lambda: DummyOptimizer())
    GREEDYMAX = Instantiable(lambda: GreedyMaxOptimizer())
