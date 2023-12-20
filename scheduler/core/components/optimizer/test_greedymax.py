# Copyright (c) 2016-2023 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from .greedymax import GreedyMaxOptimizer


def test_non_zero_intervals():
    scores = np.array([0., 1., 3., 4., 0., 0., 2., 9., 0.])
    intervals = GreedyMaxOptimizer.non_zero_intervals(scores)
    expected = np.array([[1, 4], [6, 8]], dtype=int)
    print(intervals)
    assert (expected == intervals).all()

