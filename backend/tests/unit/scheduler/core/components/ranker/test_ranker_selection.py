# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""Selecting a Ranker per run, which the Validation UI does.

Two things are worth pinning: the name -> class lookup accepts both the GraphQL enum and the
config string, and a per-run choice is REJECTED outside VALIDATION rather than ignored. The
real-time engine hardcodes DefaultRanker, so silently dropping the choice there would look
like it worked while the plans were scored by something else.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from scheduler.core.builder.modes import SchedulerModes
from scheduler.core.components.ranker import (AdditiveRanker, DefaultRanker, Ranker,
                                              RankerName, ranker_class)
from scheduler.engine.params import SchedulerParameters

_START = datetime.fromisoformat('2018-10-01 08:00:00').replace(tzinfo=ZoneInfo('UTC'))
_END = datetime.fromisoformat('2018-10-03 08:00:00').replace(tzinfo=ZoneInfo('UTC'))


def _params(**kwargs) -> SchedulerParameters:
    # semester_visibility=False + an explicit end keeps __post_init__ off the semester path.
    return SchedulerParameters(start=_START, end=_END, semester_visibility=False,
                               num_nights_to_schedule=1, **kwargs)


@pytest.mark.parametrize('name, expected', [
    (RankerName.DEFAULT, DefaultRanker),
    (RankerName.ADDITIVE, AdditiveRanker),
    # config.ranker.name arrives as a string, and OmegaConf does not normalize case.
    ('DEFAULT', DefaultRanker),
    ('ADDITIVE', AdditiveRanker),
    ('additive', AdditiveRanker),
    ('Default', DefaultRanker),
])
def test_ranker_class_resolves_enums_and_strings(name, expected):
    assert ranker_class(name) is expected


@pytest.mark.parametrize('name', ['', 'GREEDY', 'default ', None])
def test_ranker_class_rejects_unknown_names(name):
    """engine.build() turns this KeyError into ConfigurationError."""
    with pytest.raises(KeyError):
        ranker_class(name)


def test_every_ranker_name_resolves():
    """A new RankerName added without a registry entry must not slip through."""
    for name in RankerName:
        assert issubclass(ranker_class(name), Ranker)


def test_validation_accepts_a_per_run_ranker():
    assert _params(mode=SchedulerModes.VALIDATION,
                   ranker=RankerName.ADDITIVE).ranker is RankerName.ADDITIVE


@pytest.mark.parametrize('mode', [SchedulerModes.VALIDATION, SchedulerModes.SIMULATION,
                                  SchedulerModes.OPERATION])
def test_omitting_the_ranker_is_allowed_in_every_mode(mode):
    """None means 'defer to config.ranker.name', so it must never be rejected."""
    assert _params(mode=mode).ranker is None


@pytest.mark.parametrize('mode', [SchedulerModes.SIMULATION, SchedulerModes.OPERATION])
def test_a_per_run_ranker_is_rejected_outside_validation(mode):
    with pytest.raises(ValueError, match='only be selected per run in VALIDATION'):
        _params(mode=mode, ranker=RankerName.ADDITIVE)
