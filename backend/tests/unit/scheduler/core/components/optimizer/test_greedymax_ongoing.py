# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""GreedyMax places an observation that is already executing first in its site's plan.

A replan while a site is executing used to start GreedyMax at the event's slot with nothing
there, so the stitched timeline cut the running visit off. place_ongoing fills that slot with
the rest of the observation's sequence before the greedy loop runs.
"""

from datetime import datetime, timedelta, UTC
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from lucupy.minimodel import ObservationClass, ObservationID, ObservationStatus, Site
from lucupy.types import ZeroTime

from scheduler.core.components.optimizer.greedymax import GreedyMaxOptimizer
from scheduler.core.components.optimizer.timeline import Timeline

_SITE = Site.GN
_LABEL = 'G-2026B-0001-Q-0001'
_START_SLOT = 5


def _observation(label=_LABEL, cumulative=(ZeroTime, timedelta(minutes=10), timedelta(minutes=20))):
    """An ONGOING observation whose first atom is done: 10 then 20 minutes left, plus 5 of acquisition."""
    return SimpleNamespace(
        id=ObservationID(label),
        site=_SITE,
        status=ObservationStatus.ONGOING,
        obs_class=ObservationClass.SCIENCE,
        acq_overhead=timedelta(minutes=5),
        to_unique_group_id=f'group-{label}',
        cumulative_exec_times=lambda: list(cumulative),
        sequence=[
            SimpleNamespace(prog_time=timedelta(minutes=10), part_time=ZeroTime, program_used=ZeroTime)
            for _ in cumulative
        ],
    )


def _optimizer(observation, total_slots=100):
    group_data = SimpleNamespace(
        group=SimpleNamespace(observations=lambda: [observation], number_observed=0),
        group_info=SimpleNamespace(scores={0: np.ones(total_slots)}),
    )
    optimizer = GreedyMaxOptimizer()
    optimizer.selection = SimpleNamespace(night_indices=np.array([0]), starting_time_slots={_SITE: {0: _START_SLOT}})
    optimizer.sites = frozenset({_SITE})
    optimizer.time_slot_length = timedelta(minutes=1)
    optimizer.timelines = {
        0: {
            _SITE: Timeline(
                start=datetime(2026, 10, 30, tzinfo=UTC),
                end=datetime(2026, 10, 30, 10, tzinfo=UTC),
                time_slot_length=timedelta(minutes=1),
                site=_SITE,
                total_time_slots=total_slots,
            )
        }
    }
    optimizer.obs_in_plan = {_SITE: {}}
    optimizer.group_data_list = [group_data]
    optimizer.obs_group_ids = [observation.to_unique_group_id]
    return optimizer, group_data


def test_executing_observation_fills_the_start_with_the_rest_of_its_sequence():
    observation = _observation()
    optimizer, group_data = _optimizer(observation)

    with patch.object(GreedyMaxOptimizer, '_update_group_list') as update_group_list:
        optimizer.place_ongoing({_SITE: ObservationID(_LABEL)})

    slots = optimizer.timelines[0][_SITE].time_slots
    # 5 minutes of acquisition and 20 of the atoms left, starting at the plan's start.
    assert (slots[_START_SLOT : _START_SLOT + 25] == 0).all()
    assert (slots[:_START_SLOT] == Timeline.EMPTY).all()
    assert (slots[_START_SLOT + 25 :] == Timeline.EMPTY).all()

    placed = optimizer.obs_in_plan[_SITE][_START_SLOT]
    assert (placed.atom_start, placed.atom_end, placed.obs_len) == (1, 2, 25)

    # Charged and handed to the group bookkeeping, so the greedy loop does not schedule it again.
    assert observation.status is ObservationStatus.OBSERVED
    assert group_data.group.number_observed == 1
    update_group_list.assert_called_once()


def test_night_ending_first_keeps_only_the_atoms_that_fit():
    observation = _observation()
    optimizer, _ = _optimizer(observation, total_slots=_START_SLOT + 15)

    with patch.object(GreedyMaxOptimizer, '_update_group_list'):
        optimizer.place_ongoing({_SITE: ObservationID(_LABEL)})

    placed = optimizer.obs_in_plan[_SITE][_START_SLOT]
    assert (placed.atom_start, placed.atom_end, placed.obs_len) == (1, 1, 15)


def test_unschedulable_executing_observation_leaves_the_plan_alone():
    optimizer, _ = _optimizer(_observation())

    with patch.object(GreedyMaxOptimizer, '_update_group_list') as update_group_list:
        optimizer.place_ongoing({_SITE: ObservationID('G-2026B-0009-Q-0001')})

    assert (optimizer.timelines[0][_SITE].time_slots == Timeline.EMPTY).all()
    update_group_list.assert_not_called()


def test_executing_observation_with_nothing_left_is_not_placed():
    optimizer, _ = _optimizer(_observation(cumulative=(ZeroTime, ZeroTime)))

    with patch.object(GreedyMaxOptimizer, '_update_group_list'):
        optimizer.place_ongoing({_SITE: ObservationID(_LABEL)})

    assert optimizer.obs_in_plan[_SITE] == {}
