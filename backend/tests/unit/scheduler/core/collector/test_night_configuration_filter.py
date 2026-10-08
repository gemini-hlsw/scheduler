# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Only the scheduled nights are filtered over every observation.

The remaining nights of the visibility period only feed the remaining visibility of the
observations that can be scheduled, so they are filtered over that subset alone.
"""

from collections import namedtuple
from types import SimpleNamespace
from unittest.mock import MagicMock

from lucupy.minimodel import NightIndex, Site, TooType

from scheduler.core.components.collector import Collector

_SITE = Site.GN
_INSTALLED = 'GMOS-N'
_MISSING = 'IGRINS-2'

# Hashable and compared by value, like the real resources.
_Resource = namedtuple('_Resource', 'id type')


def _obs(obs_id: str, instrument: str) -> MagicMock:
    resource = _Resource(instrument, None)
    obs = MagicMock()
    obs.id = obs_id
    obs.site = _SITE
    obs.too_type = TooType.STANDARD
    obs.instrument.return_value = SimpleNamespace(id=instrument)
    obs.required_resources.return_value = [resource]
    return obs


def _night(*installed: str) -> SimpleNamespace:
    return SimpleNamespace(resources={_Resource(r, None) for r in installed},
                           filter=SimpleNamespace(program_filter=lambda _p: True))


def _collector(num_of_nights: int, nights: list) -> tuple[Collector, dict]:
    collector = Collector.__new__(Collector)
    collector.num_of_nights = num_of_nights
    collector.num_nights_calculated = len(nights)
    collector.get_program = MagicMock()
    nc = {_SITE: {NightIndex(i): night for i, night in enumerate(nights)}}
    return collector, nc


def _ids(result: dict) -> dict:
    return {int(n): sorted(o.id for o in obs) for n, obs in result.items()}


def test_observations_not_schedulable_tonight_are_skipped_on_later_nights():
    tonight = _obs('tonight', _INSTALLED)
    later_only = _obs('later-only', _MISSING)
    collector, nc = _collector(1, [_night(_INSTALLED), _night(_INSTALLED, _MISSING), _night(_INSTALLED, _MISSING)])

    result = collector._filter_by_night_configuration([('p', tonight), ('p', later_only)], nc)

    assert _ids(result) == {0: ['tonight'], 1: ['tonight'], 2: ['tonight']}


def test_later_nights_still_gate_the_schedulable_observations():
    tonight = _obs('tonight', _INSTALLED)
    collector, nc = _collector(1, [_night(_INSTALLED), _night(), _night(_INSTALLED)])

    result = collector._filter_by_night_configuration([('p', tonight)], nc)

    # Night 1 lacks the instrument, so it must not count towards the remaining visibility.
    assert _ids(result) == {0: ['tonight'], 1: [], 2: ['tonight']}


def test_every_scheduled_night_contributes_candidates():
    first = _obs('first', _INSTALLED)
    second = _obs('second', _MISSING)
    neither = _obs('neither', 'GNIRS')
    collector, nc = _collector(2, [_night(_INSTALLED), _night(_MISSING), _night(_INSTALLED, _MISSING, 'GNIRS')])

    result = collector._filter_by_night_configuration([('p', first), ('p', second), ('p', neither)], nc)

    assert _ids(result) == {0: ['first'], 1: ['second'], 2: ['first', 'second']}


def test_all_nights_scheduled_filters_every_observation():
    obs = [_obs('a', _INSTALLED), _obs('b', _MISSING)]
    collector, nc = _collector(5, [_night(_INSTALLED), _night(_MISSING)])

    result = collector._filter_by_night_configuration([('p', o) for o in obs], nc)

    assert _ids(result) == {0: ['a'], 1: ['b']}
