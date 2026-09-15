# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import numpy.typing as npt

from .base import Ranker

__all__ = [
    'DefaultRanker',
]


class DefaultRanker(Ranker):
    """
    The original, MULTIPLICATIVE Ranker: the three score terms are combined as a product of
    powers, so met_power / vis_power / wha_power act as EXPONENTS. A power of 1.0 uses the
    term as-is; 0.0 removes it (term -> 1.0). Because the terms multiply, any term that
    reaches 0 zeroes the whole score - for instance a time slot outside the hour angle
    window, where wha is 0.

    Contrast AdditiveRanker, which reads the same three fields as linear weights. Everything
    but the one expression below is shared and lives in Ranker.
    """

    def _combine_score_terms(self,
                             scale_factor: float,
                             metric: float,
                             vis_frac: float,
                             wha: npt.NDArray[float]) -> npt.NDArray[float]:
        """metric**met_power * vis_frac**vis_power * wha**wha_power (product of powers)."""
        return scale_factor * (metric ** self.params.met_power) * \
                              (vis_frac ** self.params.vis_power) * \
                              (wha ** self.params.wha_power)
