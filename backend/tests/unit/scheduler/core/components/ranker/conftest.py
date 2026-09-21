# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Fakes for the Ranker tests: no Collector, no DB, no network.

The Ranker only ever reaches into the Collector for three things, so a stub covers it:
`get_night_events(site).times[night_idx]`, `get_target_info(obs.id)` and `get_program(id)`.

The fakes must hand back *real* astropy types wherever the scoring code does unit math.
`wha` is built as `c[1] * ha / u.hourangle + c[2] / u.hourangle**2 * ha**2`, so `hourangle`
has to be an `Angle` in hourangle units or the division leaves a stray unit behind; `alt`
and `coord.dec` are compared against `Angle` altitude limits and the site latitude.
`airmass` is a plain ndarray, as it is in production.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from types import SimpleNamespace
from typing import Dict, FrozenSet, List, Optional, Tuple

import astropy.units as u
import numpy as np
import numpy.typing as npt
import pytest
from astropy.coordinates import Angle
from lucupy.minimodel import Band, NightIndex, ObservationStatus, Priority, Site

# One site and two nights everywhere, so the multi-night cases that pin the program-priority
# behaviour need no special setup.
SITE = Site.GS
NIGHTS: Tuple[NightIndex, ...] = (NightIndex(0), NightIndex(1))
N_SLOTS = 8


def _target_info(dec_deg: float = -30.0,
                 rem_visibility_frac: float = 0.42,
                 alt_deg: Optional[npt.NDArray[float]] = None,
                 visibility_slot_idx: Optional[npt.NDArray[int]] = None) -> SimpleNamespace:
    # Symmetric about the meridian so the quadratic wha weighting is exercised on both
    # sides. The default coefficients give wha = 3 - 0.08*ha**2, which crosses zero at
    # |ha| = 6.12h, so a +/-8h span makes the outer slot on each side clamp to 0.
    ha = Angle(np.linspace(-8.0, 8.0, N_SLOTS), unit=u.hourangle)
    alt = Angle(np.full(N_SLOTS, 55.0) if alt_deg is None else alt_deg, unit=u.deg)
    return SimpleNamespace(
        coord=SimpleNamespace(dec=Angle(np.full(N_SLOTS, dec_deg), unit=u.deg)),
        hourangle=ha,
        airmass=np.linspace(1.05, 1.8, N_SLOTS),
        alt=alt,
        rem_visibility_frac=rem_visibility_frac,
        visibility_slot_idx=(np.arange(N_SLOTS) if visibility_slot_idx is None
                             else visibility_slot_idx),
    )


@dataclass
class FakeCollector:
    """Only the three accessors the Ranker actually calls."""
    target_info: Dict[NightIndex, SimpleNamespace] = field(default_factory=dict)
    program: object = None
    n_slots: int = N_SLOTS
    sites: FrozenSet[Site] = frozenset({SITE})

    def get_night_events(self, site: Site) -> SimpleNamespace:
        return SimpleNamespace(times={night_idx: np.zeros(self.n_slots) for night_idx in NIGHTS})

    def get_target_info(self, obs_id) -> Optional[Dict[NightIndex, SimpleNamespace]]:
        return self.target_info

    def get_program(self, program_id) -> object:
        return self.program


@dataclass
class FakeProgram:
    thesis: bool = False
    used: timedelta = timedelta(hours=2)
    awarded: timedelta = timedelta(hours=10)
    priority_mean: float = 1.0

    def total_used(self, band: Band = None) -> timedelta:
        return self.used

    def total_awarded(self, band: Band = None) -> timedelta:
        return self.awarded

    def mean_priority(self) -> float:
        return self.priority_mean


@dataclass
class FakeObservation:
    site: Site = SITE
    band: Band = Band.BAND2
    preimaging: bool = False
    priority: Priority = Priority.MEDIUM
    status: ObservationStatus = ObservationStatus.READY
    exec_: timedelta = timedelta(hours=1)
    used: timedelta = timedelta(0)

    def __post_init__(self):
        self.id = SimpleNamespace(id='P-1[0]', program_id=lambda: 'P-1')

    def exec_time(self) -> timedelta:
        return self.exec_

    def total_used(self) -> timedelta:
        return self.used


def night_configurations(program_priority_nights: Tuple[NightIndex, ...] = ()) -> dict:
    """`night_configurations[site][night_idx].filter.program_priority_filter_any(program)`.

    Pass the nights whose calendar marks the program as priority (PV, classical, ...).
    """
    return {SITE: {night_idx: SimpleNamespace(filter=SimpleNamespace(
        program_priority_filter_any=lambda _program, _n=night_idx: _n in program_priority_nights))
        for night_idx in NIGHTS}}


@dataclass
class FakeGroup:
    """Enough of a Group for score_group / _score_and_group."""
    children_ids: Tuple[str, ...] = ('a', 'b')
    group_sites: FrozenSet[Site] = frozenset({SITE})
    and_group: bool = True
    group_name: str = 'G'

    def __post_init__(self):
        self.children = [SimpleNamespace(unique_id=cid) for cid in self.children_ids]

    def sites(self) -> FrozenSet[Site]:
        return self.group_sites

    def is_and_group(self) -> bool:
        return self.and_group

    def is_or_group(self) -> bool:
        return not self.and_group


def group_data_map(scores_by_child: Dict[str, Dict[NightIndex, List[float]]]) -> dict:
    return {cid: SimpleNamespace(group_info=SimpleNamespace(
        scores={n: np.asarray(s, dtype=float) for n, s in per_night.items()}))
        for cid, per_night in scores_by_child.items()}


@pytest.fixture
def collector() -> FakeCollector:
    """A collector whose target is visible on both nights."""
    return FakeCollector(target_info={n: _target_info() for n in NIGHTS},
                         program=FakeProgram())


@pytest.fixture
def make_target_info():
    return _target_info
