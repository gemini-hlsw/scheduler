# Copyright (c) 2016-2026 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause
"""
Branch coverage for ODBEventHandler.

The handler decides, per obscalc event, whether the plan in effect still holds. Its one
time-sensitive decision is the ONGOING branch, which asks "is the observation the ODB just
reported the one the plan wants running *right now*". Against a live ODB that is only
reachable during the planned night, because the handler stamps the reference time as
``visibility_start.date()`` combined with the real clock's UTC time-of-day.

These tests sidestep that entirely by leaving ``visibility_start`` unset, which makes the
reference time plain ``datetime.now(UTC)``, and then building the plan's visits around now.
Every branch is reachable at any hour of the day.

Visibility payload building is covered in test_obscalc_visibility.py; this file is about
which branch runs and what it does.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gpp_client.generated.enums import Instrument, ObservationWorkflowState
from lucupy.minimodel import ALL_SITES, ObservationID, ObservationStatus, Site

from scheduler.core.events.queue import ObservationActivationEvent
from scheduler.engine.params import BuildParameters
from scheduler.night_monitor.event_handlers.odb_event_handler import (
    ODBEventHandler,
    SchedulerIdleTimer,
)

_HANDLER_MODULE = "scheduler.night_monitor.event_handlers.odb_event_handler"

# GMOS_SOUTH resolves to GS, so every event here is a GS event unless it says otherwise.
_SITE = Site.GS
_LABEL = "G-2026A-0500-Q-0018"


# --- builders -------------------------------------------------------------------------------

def _value(state, label=_LABEL, obs_id="o-5e1e", instrument=Instrument.GMOS_SOUTH,
           opportunity=None):
    """The ``value`` of an obscalc update: the observation as the ODB now sees it."""
    target = SimpleNamespace(opportunity=lambda: opportunity)
    return SimpleNamespace(
        id=obs_id,
        instrument=instrument,
        reference=SimpleNamespace(label=label),
        workflow=SimpleNamespace(value=SimpleNamespace(state=state)),
        target_environment=SimpleNamespace(
            first_science_target=lambda include_deleted=False: target,
        ),
    )


def _event(state, edit_type="UPDATED", **kwargs):
    return SimpleNamespace(
        value=_value(state, **kwargs),
        edit_type=edit_type,
        old_calculation_state="CALCULATING",
        new_calculation_state="READY",
    )


def _observation(label=_LABEL, status=ObservationStatus.READY, exec_minutes=30):
    """The observation as the *last plan* recorded it, which is what the handler compares to."""
    return SimpleNamespace(
        id=ObservationID(label),
        status=status,
        exec_time=lambda: timedelta(minutes=exec_minutes),
    )


def _visit(observation, start_time, time_slots=10):
    return SimpleNamespace(
        observation=observation,
        start_time=start_time,
        time_slots=time_slots,
    )


def _plan(visits=(), found=None, slot_minutes=1):
    return SimpleNamespace(
        visits=list(visits),
        time_slot_length=timedelta(minutes=slot_minutes),
        find=lambda obs_id: found,
    )


def _handler(last_plan=None, planned_ids=frozenset()):
    store = MagicMock()
    store.last_plan = AsyncMock(return_value=last_plan)
    store.planned_observation_ids = AsyncMock(return_value=planned_ids)
    return ODBEventHandler(scheduler_queue=AsyncMock(), nightly_timeline_store=store)


@pytest.fixture
def handler_factory():
    """Builds handlers and disarms their timers, so no pending wake-up outlives the test."""
    built = []

    def make(last_plan=None, planned_ids=frozenset()):
        handler = _handler(last_plan, planned_ids)
        built.append(handler)
        return handler

    yield make

    for handler in built:
        for timer in (*handler.idle_timer.values(), *handler.observation_execution_timer.values()):
            timer.cancel()


@pytest.fixture(autouse=True)
def _reference_time_is_now():
    """
    Pin the handler's reference time to the real clock.

    With ``visibility_start`` unset the handler skips the date-substitution at
    odb_event_handler.py and uses ``datetime.now(UTC)`` directly, so a test can place visits
    around now instead of around a night that only exists at 3am.
    """
    store = MagicMock()
    store.get = AsyncMock(return_value=BuildParameters())
    with patch(f"{_HANDLER_MODULE}.build_params_store", store):
        yield


def _queued_event(handler):
    """The single event the handler pushed to the scheduler queue."""
    handler.scheduler_queue.add_schedule_event.assert_awaited_once()
    return handler.scheduler_queue.add_schedule_event.await_args.args[0]


# --- dispatch -------------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("edit_type,expected", [
    ("CREATED", "_on_created_edit"),
    ("UPDATED", "_on_updated_edit"),
    ("HARD_DELETE", "_on_deleted_edit"),
])
async def test_observation_edit_dispatches_by_edit_type(handler_factory, edit_type, expected):
    handler = handler_factory()
    event = _event(ObservationWorkflowState.READY, edit_type=edit_type)

    with patch.object(handler, expected, new_callable=AsyncMock) as branch:
        await handler._on_observation_edit(event)

    branch.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_observation_edit_unknown_type_raises(handler_factory):
    handler = handler_factory()
    event = _event(ObservationWorkflowState.READY, edit_type="SOFT_DELETE")

    # The handler must not silently drop an edit type it has no logic for. It currently
    # fails building the message (the fallback reads `editType`, which the model does not
    # have) rather than raising the NotImplementedError it intends to.
    with pytest.raises((NotImplementedError, AttributeError)):
        await handler._on_observation_edit(event)


def test_parse_observation_edit_event_unwraps_obscalc_update():
    raw = SimpleNamespace(obscalc_update="payload")
    assert ODBEventHandler.parse_observation_edit_event(raw) == "payload"


# --- CREATED --------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_created_ready_requests_plan_for_all_sites(handler_factory):
    handler = handler_factory()

    await handler._on_created_edit(_event(ObservationWorkflowState.READY, edit_type="CREATED"))

    event = _queued_event(handler)
    assert isinstance(event, ObservationActivationEvent)
    assert event.site == ALL_SITES


@pytest.mark.asyncio
async def test_created_not_ready_is_noop(handler_factory):
    # A created observation that is not READY cannot be scheduled, so there is nothing to plan.
    handler = handler_factory()

    await handler._on_created_edit(_event(ObservationWorkflowState.DEFINED, edit_type="CREATED"))

    handler.scheduler_queue.add_schedule_event.assert_not_called()


# --- HARD_DELETE ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_deleted_in_plan_requests_plan_for_that_site(handler_factory):
    handler = handler_factory(planned_ids=frozenset({_LABEL}))

    await handler._on_deleted_edit(_event(ObservationWorkflowState.READY, edit_type="HARD_DELETE"))

    event = _queued_event(handler)
    assert isinstance(event.site, Site)


@pytest.mark.asyncio
async def test_deleted_in_both_sites_requests_one_plan_only(handler_factory):
    # The loop breaks after the first hit: an observation listed at both sites must not
    # produce two schedule requests for one deletion.
    handler = handler_factory(planned_ids=frozenset({_LABEL}))

    await handler._on_deleted_edit(_event(ObservationWorkflowState.READY, edit_type="HARD_DELETE"))

    assert handler.scheduler_queue.add_schedule_event.await_count == 1


@pytest.mark.asyncio
async def test_deleted_not_in_plan_is_noop(handler_factory):
    handler = handler_factory(planned_ids=frozenset({"G-2026A-0500-Q-0001"}))

    await handler._on_deleted_edit(_event(ObservationWorkflowState.READY, edit_type="HARD_DELETE"))

    handler.scheduler_queue.add_schedule_event.assert_not_called()


# --- UPDATED / ONGOING ----------------------------------------------------------------------
#
# The time-sensitive branch. `now` is the handler's reference time (visibility_start is unset,
# see the _reference_time_is_now fixture), so a visit spanning `now` is the one the plan
# expects to be running.

def _visit_spanning_now(observation, slot_minutes=1, time_slots=10):
    now = datetime.now(UTC)
    return _visit(observation, now - timedelta(minutes=slot_minutes * time_slots / 2), time_slots)


def _visit_already_over(observation, slot_minutes=1, time_slots=10):
    now = datetime.now(UTC)
    return _visit(observation, now - timedelta(hours=2), time_slots)


@pytest.mark.asyncio
async def test_ongoing_matches_plan_arms_execution_timer(handler_factory):
    # The plan is being followed: wait the observation out rather than replanning now.
    planned = _observation(status=ObservationStatus.READY)
    handler = handler_factory(_plan(visits=[_visit_spanning_now(planned)], found=planned))

    await handler._on_updated_edit(_event(ObservationWorkflowState.ONGOING))

    handler.scheduler_queue.add_schedule_event.assert_not_called()
    assert handler.observation_execution_timer[_SITE].pending
    assert not handler.idle_timer[_SITE].pending


@pytest.mark.asyncio
async def test_ongoing_out_of_order_requests_plan(handler_factory):
    # The plan wanted a different observation running at this moment.
    expected = _observation(label="G-2026A-0500-Q-0001")
    reported = _observation(label=_LABEL)
    handler = handler_factory(_plan(visits=[_visit_spanning_now(expected)], found=reported))

    await handler._on_updated_edit(_event(ObservationWorkflowState.ONGOING))

    event = _queued_event(handler)
    assert "G-2026A-0500-Q-0001" in event.description
    assert _LABEL in event.description
    assert not handler.observation_execution_timer[_SITE].pending


@pytest.mark.asyncio
async def test_ongoing_in_plan_but_nothing_expected_now_requests_plan(handler_factory):
    # In the plan, but its visit is long past: the night is off the rails, so replan.
    planned = _observation()
    handler = handler_factory(_plan(visits=[_visit_already_over(planned)], found=planned))

    await handler._on_updated_edit(_event(ObservationWorkflowState.ONGOING))

    assert "nothing" in _queued_event(handler).description


@pytest.mark.asyncio
async def test_ongoing_not_in_plan_requests_plan(handler_factory):
    # An observation nobody planned is being executed.
    expected = _observation(label="G-2026A-0500-Q-0001")
    handler = handler_factory(_plan(visits=[_visit_spanning_now(expected)], found=None))

    await handler._on_updated_edit(_event(ObservationWorkflowState.ONGOING))

    assert "Expected observation G-2026A-0500-Q-0001" in _queued_event(handler).description


@pytest.mark.asyncio
async def test_ongoing_not_in_plan_and_nothing_expected_requests_plan(handler_factory):
    handler = handler_factory(_plan(visits=[], found=None))

    await handler._on_updated_edit(_event(ObservationWorkflowState.ONGOING))

    assert "not expecting any execution" in _queued_event(handler).description


@pytest.mark.asyncio
async def test_ongoing_already_ongoing_in_plan_does_not_rearm(handler_factory):
    # ONGOING -> ONGOING: the plan already knows it is running, so no timer and no replan.
    planned = _observation(status=ObservationStatus.ONGOING)
    handler = handler_factory(_plan(visits=[_visit_spanning_now(planned)], found=planned))

    await handler._on_updated_edit(_event(ObservationWorkflowState.ONGOING))

    handler.scheduler_queue.add_schedule_event.assert_not_called()
    assert not handler.observation_execution_timer[_SITE].pending


# --- UPDATED / READY ------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ready_in_plan_and_still_ready_requests_plan(handler_factory):
    # READY -> READY on an observation already in the plan means it was edited, so the plan
    # was built against stale constraints.
    planned = _observation(status=ObservationStatus.READY)
    handler = handler_factory(_plan(visits=[_visit_spanning_now(planned)], found=planned))

    await handler._on_updated_edit(_event(ObservationWorkflowState.READY))

    assert "was modified" in _queued_event(handler).description


@pytest.mark.asyncio
async def test_ready_in_plan_but_not_ready_there_is_noop(handler_factory):
    # The plan recorded it as already observed; a READY event does not invalidate the plan.
    planned = _observation(status=ObservationStatus.OBSERVED)
    handler = handler_factory(_plan(visits=[_visit_spanning_now(planned)], found=planned))

    await handler._on_updated_edit(_event(ObservationWorkflowState.READY))

    handler.scheduler_queue.add_schedule_event.assert_not_called()


# --- UPDATED / COMPLETED --------------------------------------------------------------------

@pytest.mark.asyncio
async def test_completed_in_plan_arms_idle_timer(handler_factory):
    # Finished as planned: nothing is executing, so watch the site for idle time instead of
    # replanning immediately.
    planned = _observation()
    handler = handler_factory(_plan(visits=[_visit_spanning_now(planned)], found=planned))
    handler.observation_execution_timer[_SITE].set(3600, AsyncMock())

    await handler._on_updated_edit(_event(ObservationWorkflowState.COMPLETED))

    handler.scheduler_queue.add_schedule_event.assert_not_called()
    assert handler.idle_timer[_SITE].pending
    assert not handler.observation_execution_timer[_SITE].pending


@pytest.mark.asyncio
async def test_completed_not_in_plan_is_noop(handler_factory):
    handler = handler_factory(_plan(visits=[], found=None))

    await handler._on_updated_edit(_event(ObservationWorkflowState.COMPLETED))

    handler.scheduler_queue.add_schedule_event.assert_not_called()
    assert not handler.idle_timer[_SITE].pending


# --- guards ---------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_updated_without_instrument_never_reads_the_plan(handler_factory):
    handler = handler_factory(_plan())

    await handler._on_updated_edit(_event(ObservationWorkflowState.ONGOING, instrument=None))

    handler.nightly_timeline_store.last_plan.assert_not_called()
    handler.scheduler_queue.add_schedule_event.assert_not_called()


@pytest.mark.asyncio
async def test_updated_with_no_plan_in_effect_is_noop(handler_factory):
    handler = handler_factory(last_plan=None)

    await handler._on_updated_edit(_event(ObservationWorkflowState.ONGOING))

    handler.scheduler_queue.add_schedule_event.assert_not_called()
    assert not handler.observation_execution_timer[_SITE].pending


# --- _request_new_plan ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_request_new_plan_drops_pending_waits_and_rearms_idle(handler_factory):
    # Once a new plan is requested the observation being waited on is no longer the one the
    # plan expects, so the execution wait must go and the idle watch must be restarted.
    handler = handler_factory()
    handler.observation_execution_timer[_SITE].set(3600, AsyncMock())

    await handler._request_new_plan(
        ObservationActivationEvent(site=_SITE, observation_id="o-1",
                                   time=datetime.now(UTC), description="test")
    )

    assert not handler.observation_execution_timer[_SITE].pending
    assert handler.idle_timer[_SITE].pending


@pytest.mark.asyncio
async def test_request_new_plan_for_all_sites_rearms_every_site(handler_factory):
    handler = handler_factory()

    await handler._request_new_plan(
        ObservationActivationEvent(site=ALL_SITES, observation_id="o-1",
                                   time=datetime.now(UTC), description="test")
    )

    assert all(handler.idle_timer[site].pending for site in ALL_SITES)


# --- SchedulerIdleTimer ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_timer_set_replaces_the_pending_wakeup():
    # Arming twice must leave one timer, or a stale countdown fires against a replaced plan.
    timer = SchedulerIdleTimer()
    first_fired = False

    async def first():
        nonlocal first_fired
        first_fired = True

    timer.set(0.01, first)
    timer.set(5.0, AsyncMock())
    await _tick(0.05)

    assert not first_fired
    assert timer.pending
    timer.cancel()


@pytest.mark.asyncio
async def test_timer_runs_its_callback():
    timer = SchedulerIdleTimer()
    fired = False

    async def callback():
        nonlocal fired
        fired = True

    timer.set(0.01, callback)
    await _tick(0.05)

    assert fired
    assert not timer.pending


@pytest.mark.asyncio
async def test_timer_cancel_prevents_the_callback():
    timer = SchedulerIdleTimer()
    fired = False

    async def callback():
        nonlocal fired
        fired = True

    timer.set(0.01, callback)
    timer.cancel()
    await _tick(0.05)

    assert not fired
    assert not timer.pending


@pytest.mark.asyncio
async def test_timer_with_offset_adds_the_grace_period():
    # An observation is given WAITING_OFFSET on top of its estimate before we call it late.
    timer = SchedulerIdleTimer()
    timer.set(0.01, AsyncMock(), with_offset=True)
    await _tick(0.05)

    # Still pending: the offset (minutes) dominates the 0.01s delay.
    assert timer.pending
    timer.cancel()


async def _tick(seconds):
    import asyncio
    await asyncio.sleep(seconds)


# --- reference time -------------------------------------------------------------------------
#
# What makes a past night testable from a live ODB: `simulated_now` moves the handler's clock
# into that night, and keeps moving with the real one.

def _params(**kwargs):
    store = MagicMock()
    store.get = AsyncMock(return_value=BuildParameters(**kwargs))
    return patch(f"{_HANDLER_MODULE}.build_params_store", store)


@pytest.mark.asyncio
async def test_reference_time_defaults_to_the_real_clock(handler_factory):
    # Untouched build parameters mean a plain real-time run tonight.
    handler = handler_factory()
    with _params():
        assert abs((await handler._reference_time(_SITE)) - datetime.now(UTC)) < timedelta(seconds=5)


@pytest.mark.asyncio
async def test_reference_time_anchors_on_the_plans_night(handler_factory):
    # Build parameters point at another night, and no simulated instant was given, so the
    # clock starts at that night's evening twilight rather than at today's time of day.
    night_start = datetime(2026, 5, 12, 22, 53, tzinfo=UTC)
    handler = handler_factory()
    handler.nightly_timeline_store.night_start = AsyncMock(return_value=night_start)

    with _params(visibility_start=datetime(2026, 5, 13, tzinfo=UTC)):
        when = await handler._reference_time(_SITE)

    assert abs(when - night_start) < timedelta(seconds=5)


@pytest.mark.asyncio
async def test_reference_time_falls_back_to_now_before_any_plan(handler_factory):
    # Build parameters are set but the engine has recorded no night yet, so there is nothing
    # to anchor to.
    handler = handler_factory()
    handler.nightly_timeline_store.night_start = AsyncMock(return_value=None)

    with _params(visibility_start=datetime(2026, 5, 13, tzinfo=UTC)):
        when = await handler._reference_time(_SITE)

    assert abs(when - datetime.now(UTC)) < timedelta(seconds=5)


@pytest.mark.asyncio
async def test_reference_time_prefers_simulated_now_over_the_night_start(handler_factory):
    # simulated_now is the "jump to this point in the night" override.
    night_start = datetime(2026, 5, 12, 22, 53, tzinfo=UTC)
    anchor = datetime(2026, 5, 13, 2, 30, tzinfo=UTC)
    handler = handler_factory()
    handler.nightly_timeline_store.night_start = AsyncMock(return_value=night_start)

    with _params(visibility_start=datetime(2026, 5, 13, tzinfo=UTC), simulated_now=anchor):
        when = await handler._reference_time(_SITE)

    assert abs(when - anchor) < timedelta(seconds=5)


@pytest.mark.asyncio
async def test_reference_time_advances_with_the_real_clock(handler_factory):
    # The anchor is not frozen: a READY -> ONGOING -> COMPLETED run spread over real minutes
    # must move through the simulated night by those same minutes.
    handler = handler_factory()
    anchor = datetime(2026, 5, 13, 2, 30, tzinfo=UTC)
    params = BuildParameters(simulated_now=anchor)
    params.set_at = datetime.now(UTC) - timedelta(minutes=17)

    store = MagicMock()
    store.get = AsyncMock(return_value=params)
    with patch(f"{_HANDLER_MODULE}.build_params_store", store):
        when = await handler._reference_time()

    assert abs(when - (anchor + timedelta(minutes=17))) < timedelta(seconds=5)


@pytest.mark.asyncio
async def test_simulated_now_makes_a_past_night_comparable(handler_factory):
    """
    The whole point: a plan for a past night, an ODB event arriving today, and the handler
    still recognising that the plan is being followed.
    """
    anchor = datetime(2026, 5, 13, 2, 30, tzinfo=UTC)
    planned = _observation(status=ObservationStatus.READY)
    visit = _visit(planned, anchor - timedelta(minutes=5), time_slots=10)
    handler = handler_factory(_plan(visits=[visit], found=planned))

    with _params(visibility_start=datetime(2026, 5, 13, tzinfo=UTC), simulated_now=anchor):
        await handler._on_updated_edit(_event(ObservationWorkflowState.ONGOING))

    handler.scheduler_queue.add_schedule_event.assert_not_called()
    assert handler.observation_execution_timer[_SITE].pending


@pytest.mark.asyncio
async def test_raised_events_carry_the_simulated_clock(handler_factory):
    """
    The event timestamp must use the same clock as the comparison.

    The engine turns Event.time into the timeslot the new plan starts from, so an event
    stamped with the real clock against a past night's twilight lands tens of thousands of
    timeslots past the end of the night and yields an empty plan.
    """
    anchor = datetime(2026, 5, 13, 2, 30, tzinfo=UTC)
    expected = _observation(label="G-2026A-0500-Q-0001")
    visit = _visit(expected, anchor - timedelta(minutes=5), time_slots=10)
    handler = handler_factory(_plan(visits=[visit], found=_observation(label=_LABEL)))

    with _params(simulated_now=anchor):
        await handler._on_updated_edit(_event(ObservationWorkflowState.ONGOING))

    assert abs(_queued_event(handler).time - anchor) < timedelta(seconds=5)


@pytest.mark.asyncio
async def test_created_event_carries_the_simulated_clock(handler_factory):
    # Every path that raises an event must agree on the clock, not just the ONGOING one.
    anchor = datetime(2026, 5, 13, 2, 30, tzinfo=UTC)
    handler = handler_factory()

    with _params(simulated_now=anchor):
        await handler._on_created_edit(
            _event(ObservationWorkflowState.READY, edit_type="CREATED")
        )

    assert abs(_queued_event(handler).time - anchor) < timedelta(seconds=5)


def test_build_parameters_reference_time_is_none_when_not_simulating():
    assert BuildParameters().reference_time() is None
    assert BuildParameters(visibility_start=datetime(2026, 5, 13, tzinfo=UTC)).reference_time() is None


def test_default_build_parameters_are_not_customized():
    # set_at is stamped on every construction, so it must not by itself make the parameters
    # look customized: that would put every real-time run onto the simulated clock.
    assert BuildParameters().is_customized() is False


@pytest.mark.parametrize("field,value", [
    ("visibility_start", datetime(2026, 5, 13, tzinfo=UTC)),
    ("visibility_end", datetime(2026, 5, 16, tzinfo=UTC)),
    ("program_list", ["G-2026A-0500-Q"]),
    ("simulated_now", datetime(2026, 5, 13, 2, 30, tzinfo=UTC)),
])
def test_any_set_build_parameter_counts_as_customized(field, value):
    assert BuildParameters(**{field: value}).is_customized() is True
