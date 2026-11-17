# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""GreedyMax treats an observation being executed as done.

The plan starts once the running observation is expected to finish. Mid-run the ODB still lists
its remaining atoms and the Ranker boosts ONGOING, so without charge_ongoing the greedy loop
would schedule it again right at the start.
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
_END = datetime(2026, 11, 17, 4, 0, tzinfo=UTC)


def _observation(cumulative=(ZeroTime, timedelta(minutes=10), timedelta(minutes=20))):
    """An ONGOING observation whose first atom is done: 10 then 20 minutes left."""
    return SimpleNamespace(
        id=ObservationID(_LABEL),
        site=_SITE,
        status=ObservationStatus.ONGOING,
        obs_class=ObservationClass.SCIENCE,
        acq_overhead=timedelta(minutes=5),
        cumulative_exec_times=lambda: list(cumulative),
        sequence=[
            SimpleNamespace(prog_time=timedelta(minutes=10), part_time=ZeroTime, program_used=ZeroTime, observed=False)
            for _ in cumulative
        ],
    )


def _optimizer(observation):
    group_data = SimpleNamespace(group=SimpleNamespace(observations=lambda: [observation], number_observed=0))
    optimizer = GreedyMaxOptimizer()
    optimizer.selection = SimpleNamespace(night_indices=np.array([0]))
    optimizer.group_data_list = [group_data]
    optimizer.timelines = {
        0: {
            _SITE: Timeline(
                start=datetime(2026, 11, 17, tzinfo=UTC),
                end=datetime(2026, 11, 17, 10, tzinfo=UTC),
                time_slot_length=timedelta(minutes=1),
                site=_SITE,
                total_time_slots=100,
            )
        }
    }
    return optimizer, group_data


def test_executing_observation_is_charged_as_done():
    observation = _observation()
    optimizer, group_data = _optimizer(observation)

    with patch.object(GreedyMaxOptimizer, '_update_group_list') as update_group_list:
        optimizer.charge_ongoing({ObservationID(_LABEL): _END})

    # Every atom left is charged, so the group reads as observed.
    assert observation.status is ObservationStatus.OBSERVED
    assert not observation.sequence[0].observed
    assert all(atom.observed for atom in observation.sequence[1:])
    assert group_data.group.number_observed == 1

    # Handed to the same bookkeeping as an added group, timed from its expected end.
    [(max_group_info, night_idx)] = [call.args for call in update_group_list.call_args_list]
    assert max_group_info.group_data is group_data
    assert max_group_info.end_time == _END
    assert night_idx == 0

    # It is not placed: the plan starts after it.
    assert (optimizer.timelines[0][_SITE].time_slots == Timeline.EMPTY).all()


def test_unschedulable_executing_observation_is_left_alone():
    optimizer, group_data = _optimizer(_observation())

    with patch.object(GreedyMaxOptimizer, '_update_group_list') as update_group_list:
        optimizer.charge_ongoing({ObservationID('G-2026B-0009-Q-0001'): _END})

    assert group_data.group.number_observed == 0
    update_group_list.assert_not_called()


def test_executing_observation_with_nothing_left_is_not_charged_again():
    observation = _observation(cumulative=(ZeroTime, ZeroTime))
    optimizer, group_data = _optimizer(observation)

    with patch.object(GreedyMaxOptimizer, '_update_group_list') as update_group_list:
        optimizer.charge_ongoing({ObservationID(_LABEL): _END})

    assert observation.status is ObservationStatus.ONGOING
    assert group_data.group.number_observed == 1
    update_group_list.assert_called_once()
