# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""ExecutionTracker turns the ODB SLEW/SEQUENCE events into executions per site.

An execution begins with a START_SLEW or, when no slew is needed, a sequence START, and
ends with a sequence STOP. ABORT drops it unrecorded. Events are built with gpp-client's
own models so the timestamp and site parsing is exercised too.
"""

import asyncio
from datetime import datetime, UTC
from unittest.mock import patch

import pytest
from gpp_client.generated.scheduler_executed_events import SchedulerExecutedEvents
from lucupy.minimodel import ObservationID, Site

from scheduler.night_monitor.execution_tracker import ExecutionTracker

OBS_A = 'G-2026B-0001-Q-0001'
OBS_B = 'G-2026B-0002-Q-0001'
OBS_C = 'G-2026B-0003-Q-0001'


def _observation(label: str) -> dict:
    return {'id': f'o-{label[-4:]}', 'reference': {'label': label}, 'execution': {'digest': None}}


def _slew(label: str, stage: str, time: str, site: str = 'GN') -> SchedulerExecutedEvents:
    return SchedulerExecutedEvents.model_validate({'executionEventAdded': {'value': {
        '__typename': 'SlewEvent', 'eventType': 'SLEW', 'id': 'e-1', 'recordedTime': time,
        'slewStage': stage, 'observation': _observation(label), 'visit': {'site': site},
    }}})


def _sequence(label: str, command: str, time: str, site: str = 'GN') -> SchedulerExecutedEvents:
    return SchedulerExecutedEvents.model_validate({'executionEventAdded': {'value': {
        '__typename': 'SequenceEvent', 'eventType': 'SEQUENCE', 'id': 'e-1', 'effectiveTime': time,
        'command': command, 'observation': _observation(label), 'visit': {'site': site},
    }}})


def _at(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 10, 29, hour, minute, tzinfo=UTC)


def _odb_time(hour: int, minute: int = 0) -> str:
    # The ODB Timestamp scalar: space separated, microseconds, no zone (UTC).
    return f'2026-10-29 {hour:02}:{minute:02}:00.000000'


def _feed(tracker: ExecutionTracker, *events: SchedulerExecutedEvents) -> None:
    for event in events:
        tracker._handle_event(event.execution_event_added.value)


def _labels(records) -> list:
    return [r.obs_id.id for r in records]


def test_slew_then_sequence_is_one_execution():
    tracker = ExecutionTracker()
    _feed(
        tracker,
        _slew(OBS_A, 'START_SLEW', _odb_time(1)),
        _slew(OBS_A, 'END_SLEW', _odb_time(1, 5)),
        _sequence(OBS_A, 'START', _odb_time(1, 6)),
        _sequence(OBS_A, 'STOP', _odb_time(2)),
    )

    [record] = tracker.get_executed_observations()
    assert record.obs_id == ObservationID(OBS_A)
    assert record.site == Site.GN
    # The slew starts it; the sequence START after it is the same execution.
    assert record.start == _at(1)
    assert record.end == _at(2)
    assert not record.missing_stop
    assert not tracker.is_currently_executed(ObservationID(OBS_A))


def test_sequence_start_without_slew_starts_execution():
    tracker = ExecutionTracker()
    _feed(tracker, _sequence(OBS_A, 'START', _odb_time(1)))

    assert tracker.is_currently_executed(ObservationID(OBS_A))
    current = tracker.get_current_execution(Site.GN)
    assert current.start == _at(1)
    assert current.end is None


def test_pause_and_continue_keep_it_executing():
    tracker = ExecutionTracker()
    _feed(
        tracker,
        _sequence(OBS_A, 'START', _odb_time(1)),
        _sequence(OBS_A, 'PAUSE', _odb_time(1, 10)),
        _sequence(OBS_A, 'CONTINUE', _odb_time(1, 20)),
    )

    assert tracker.is_currently_executed(ObservationID(OBS_A))
    assert tracker.get_current_execution(Site.GN).start == _at(1)


def test_abort_drops_execution_unrecorded():
    tracker = ExecutionTracker()
    _feed(
        tracker,
        _slew(OBS_A, 'START_SLEW', _odb_time(1)),
        _sequence(OBS_A, 'ABORT', _odb_time(1, 30)),
    )

    assert not tracker.is_currently_executed(ObservationID(OBS_A))
    assert tracker.get_executed_observations() == []


def test_new_start_closes_execution_missing_its_stop():
    tracker = ExecutionTracker()
    _feed(
        tracker,
        _slew(OBS_A, 'START_SLEW', _odb_time(1)),
        _slew(OBS_B, 'START_SLEW', _odb_time(2)),
    )

    closed, running = tracker.get_executed_observations()
    assert closed.obs_id == ObservationID(OBS_A)
    assert closed.end == _at(2)
    assert closed.missing_stop
    assert running.obs_id == ObservationID(OBS_B)
    assert running.end is None


def test_sites_execute_independently():
    tracker = ExecutionTracker()
    _feed(
        tracker,
        _slew(OBS_A, 'START_SLEW', _odb_time(1), site='GN'),
        _sequence(OBS_B, 'START', _odb_time(1, 30), site='GS'),
    )

    # GS starting must not close the GN execution.
    assert tracker.get_current_execution(Site.GN).obs_id == ObservationID(OBS_A)
    assert tracker.get_current_execution(Site.GS).obs_id == ObservationID(OBS_B)
    assert not any(r.missing_stop for r in tracker.get_executed_observations())


def test_stop_without_matching_start_is_ignored():
    tracker = ExecutionTracker()
    _feed(
        tracker,
        _slew(OBS_A, 'START_SLEW', _odb_time(1)),
        _sequence(OBS_B, 'STOP', _odb_time(2)),
    )

    assert tracker.is_currently_executed(ObservationID(OBS_A))
    assert _labels(tracker.get_executed_observations()) == [OBS_A]


@pytest.mark.parametrize(
    ('from_', 'to', 'expected'),
    [
        (None, None, [OBS_A, OBS_B, OBS_C]),
        # Only A had started by 01:15.
        (None, _at(1, 15), [OBS_A]),
        # A ended at 02:00; B and C overlap the window.
        (_at(2, 5), _at(3, 30), [OBS_B, OBS_C]),
        # C is still executing, so it overlaps any later window.
        (_at(5), None, [OBS_C]),
    ],
)
def test_range_returns_overlapping_executions(from_, to, expected):
    tracker = ExecutionTracker()
    _feed(
        tracker,
        _slew(OBS_A, 'START_SLEW', _odb_time(1)),
        _sequence(OBS_A, 'STOP', _odb_time(2)),
        _sequence(OBS_B, 'START', _odb_time(1, 30), site='GS'),
        _sequence(OBS_B, 'STOP', _odb_time(3), site='GS'),
        _slew(OBS_C, 'START_SLEW', _odb_time(3, 10)),
    )

    assert _labels(tracker.get_executed_observations(from_, to)) == expected


def test_zulu_timestamp_parses_as_utc():
    tracker = ExecutionTracker()
    _feed(tracker, _sequence(OBS_A, 'START', '2026-10-29T01:00:00Z'))

    assert tracker.get_current_execution(Site.GN).start == _at(1)


@pytest.mark.asyncio
async def test_bad_event_does_not_stop_tracking():
    async def events():
        yield _sequence(OBS_A, 'START', 'not a timestamp')
        yield _sequence(OBS_B, 'START', _odb_time(1))

    tracker = ExecutionTracker()
    with patch('scheduler.night_monitor.execution_tracker.gpp') as gpp:
        gpp.client.scheduler.subscribe_to_executed_events = events
        await asyncio.wait_for(tracker.start(), timeout=1)

    assert not tracker.is_currently_executed(ObservationID(OBS_A))
    assert tracker.is_currently_executed(ObservationID(OBS_B))


@pytest.mark.asyncio
async def test_stop_clears_everything():
    tracker = ExecutionTracker()
    _feed(
        tracker,
        _slew(OBS_A, 'START_SLEW', _odb_time(1)),
        _sequence(OBS_A, 'STOP', _odb_time(2)),
        _sequence(OBS_B, 'START', _odb_time(2, 30)),
    )

    await tracker.stop()

    assert tracker.get_executed_observations() == []
    assert tracker.get_current_execution(Site.GN) is None
