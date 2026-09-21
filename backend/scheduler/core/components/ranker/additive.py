# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import numpy.typing as npt

from .base import Ranker

__all__ = [
    'AdditiveRanker',
]


class AdditiveRanker(Ranker):
    """
    Additive Ranker combine the scoring terms by addition instead of
    multiplication like the default.
    """

    def _combine_score_terms(self,
                             scale_factor: float,
                             metric: float,
                             vis_frac: float,
                             wha: npt.NDArray[float]) -> npt.NDArray[float]:
        """metric*met_power + vis_frac*vis_power + wha*wha_power (weighted sum)."""
        return scale_factor * ((metric * self.params.met_power)
                               + (vis_frac * self.params.vis_power)
                               + (wha * self.params.wha_power))
