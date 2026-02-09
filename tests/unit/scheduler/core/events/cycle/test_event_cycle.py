import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np
from lucupy.timeutils import time2slots

from scheduler.core.components.changemonitor import ChangeMonitor, TimeCoordinateRecord
from scheduler.core.events.queue import NightlyTimeline, NightEventQueue
from scheduler.core.events.cycle import EventCycle
from scheduler.core.plans import Plans


from lucupy.minimodel import TimeslotIndex, NightIndex, Site


class MockEvent:
    def __init__(self, time, description="Test Event"):
        self.time = time
        self.description = description

    def to_timeslot_idx(self, night_start, time_slot_length):
        time_from_twilight = self.time - night_start
        time_slots_from_twilight = time2slots(time_slot_length, time_from_twilight)
        return TimeslotIndex(time_slots_from_twilight)


class TestEventCycle:

    def test_init(self, setup_basic_components):
        """Test initialization of EventCycle."""
        comps = setup_basic_components

        event_cycle = EventCycle(
            params=comps['params'],
            queue=comps['queue'],
            scp=comps['scp']
        )

        assert event_cycle.params == comps['params']
        assert event_cycle.queue == comps['queue']
        assert event_cycle.scp == comps['scp']
        assert isinstance(event_cycle.change_monitor, ChangeMonitor)

    def test_process_current_events_empty_queue(self, setup_event_cycle, setup_night_events):
        """Test processing current events with an empty queue."""
        event_cycle, comps = setup_event_cycle

        site = comps['params'].sites
        night_idx = NightIndex(0)

        events_by_night = MagicMock(spec=NightEventQueue)
        events_by_night.is_empty.return_value = True
        events_by_night.has_more_events.return_value = False

        nightly_timeline = MagicMock(spec=NightlyTimeline)

        # This should raise an error since we have no morning twilight
        with pytest.raises(RuntimeError):
            event_cycle.run(site, night_idx, nightly_timeline)

    def test_process_current_first_event(self, setup_event_cycle, setup_night_events):
        """Test processing with no more events."""
        event_cycle, comps = setup_event_cycle
        night_data = setup_night_events

        site = comps['params'].sites
        night_idx = NightIndex(0)
        night_start = night_data['base_time']
        time_slot_length = timedelta(minutes=1)

        event = MockEvent(night_start, "Event")
        event_timeslot = TimeslotIndex(0)
        events_by_night = MagicMock(spec=NightEventQueue)
        events_by_night.is_empty.return_value = False
        events_by_night.has_more_events.return_value = True
        events_by_night.top_event.return_value = event

        plans = MagicMock(spec=Plans)

        update, current_timeslot = event_cycle._process_current_events(
            site, night_idx, night_start, time_slot_length,
            events_by_night, plans, None, None
        )

        # Should return the same next_event and next_event_timeslot
        assert current_timeslot == event_timeslot
        assert update is not None

    def test_process_current_events_future_event(self, setup_event_cycle, setup_night_events):
        """Test processing with an event in the future."""
        event_cycle, comps = setup_event_cycle
        night_data = setup_night_events

        site = comps['params'].sites
        night_idx = NightIndex(0)
        night_start = night_data['base_time']
        time_slot_length = timedelta(minutes=1)

        # Should be executed in 60 minutes (timeslot 60)
        future_event = MockEvent(night_start + timedelta(hours=1), "Future Event")

        events_by_night = MagicMock(spec=NightEventQueue)
        events_by_night.is_empty.return_value = False
        events_by_night.has_more_events.return_value = True
        events_by_night.top_event.return_value = future_event

        plans = MagicMock(spec=Plans)
        previous_timeslot = TimeslotIndex(0)
        update = MagicMock(spec=TimeCoordinateRecord)
        update.timeslot_idx = TimeslotIndex(10)

        update_result, current_timeslot = event_cycle._process_current_events(
            site, night_idx, night_start, time_slot_length,
            events_by_night, plans, update, previous_timeslot
        )

        # Should update next_event and next_event_timeslot
        assert current_timeslot == update.timeslot_idx
        # Should not process the event yet
        events_by_night.pop_next_event.assert_not_called()


    def test_process_current_events_future_event_sooner_timeslot(self, setup_event_cycle, setup_night_events):
        """Test processing with an event in the future."""
        event_cycle, comps = setup_event_cycle
        night_data = setup_night_events

        site = comps['params'].sites
        night_idx = NightIndex(0)
        night_start = night_data['base_time']
        time_slot_length = timedelta(minutes=1)

        # Should be executed in 60 minutes (timeslot 60)
        future_event = MockEvent(night_start + timedelta(hours=1), "Future Event")
        future_event_timeslot = TimeslotIndex(60)

        events_by_night = MagicMock(spec=NightEventQueue)
        events_by_night.is_empty.return_value = False
        events_by_night.has_more_events.return_value = True
        events_by_night.top_event.return_value = future_event

        plans = MagicMock(spec=Plans)
        previous_timeslot = TimeslotIndex(0)
        update = MagicMock(spec=TimeCoordinateRecord)
        update.timeslot_idx = TimeslotIndex(120)

        _, current_timeslot = event_cycle._process_current_events(
            site, night_idx, night_start, time_slot_length,
            events_by_night, plans, update, previous_timeslot
        )

        # Should update next_event and next_event_timeslot
        assert current_timeslot == future_event_timeslot
        # Should not process the event yet
        events_by_night.pop_next_event.assert_called_once()

    def test_handle_updates_no_update(self, setup_event_cycle):
        """Test handling updates when no update is scheduled."""
        event_cycle, comps = setup_event_cycle

        site = comps['params'].sites
        night_idx = NightIndex(0)
        current_timeslot = TimeslotIndex(0)
        update = None
        plans = MagicMock(spec=Plans)
        nightly_timeline = MagicMock(spec=NightlyTimeline)

        result_plans = event_cycle._handle_updates(
            site, night_idx, current_timeslot, update, plans, nightly_timeline
        )

        # Should not change anything
        assert result_plans == plans

    def test_handle_updates_future_update(self, setup_event_cycle):
        """Test handling updates when update is scheduled for the future."""
        event_cycle, comps = setup_event_cycle

        site = comps['params'].sites
        night_idx = NightIndex(0)
        current_timeslot = TimeslotIndex(0)

        update_event = MockEvent(datetime.now(), "Update Event")
        time_record = TimeCoordinateRecord(
            timeslot_idx=TimeslotIndex(5),  # Future timeslot
            event=update_event,
            done=False,
            perform_time_accounting=True
        )
        next_update = time_record

        plans = MagicMock(spec=Plans)
        nightly_timeline = MagicMock(spec=NightlyTimeline)

        result_plans = event_cycle._handle_updates(
            site, night_idx, current_timeslot, next_update, plans, nightly_timeline
        )

        # Should not change anything
        assert result_plans == plans

    def test_handle_updates_current_update_not_done(self, setup_event_cycle):
        """Test handling updates when update is scheduled for current timeslot and night not done."""
        event_cycle, comps = setup_event_cycle

        site = comps['params'].sites
        night_idx = NightIndex(0)
        current_timeslot = TimeslotIndex(5)

        update_event = MockEvent(datetime.now(), "Update Event")
        time_record = TimeCoordinateRecord(
            timeslot_idx=TimeslotIndex(5),
            event=update_event,
            done=False,
            perform_time_accounting=True
        )
        next_update = time_record

        plans = MagicMock(spec=Plans)
        nightly_timeline = MagicMock(spec=NightlyTimeline)
        new_plans = MagicMock(spec=Plans)

        # Mock the methods we'll call
        event_cycle._perform_time_accounting = MagicMock()
        event_cycle._create_new_plan = MagicMock(return_value=new_plans)

        result_plans = event_cycle._handle_updates(
            site, night_idx, current_timeslot, next_update, plans, nightly_timeline
        )

        # Should perform time accounting and create new plan
        event_cycle._perform_time_accounting.assert_called_once()
        event_cycle._create_new_plan.assert_called_once()

        # Should update plans and clear next_update
        assert result_plans == new_plans

    def test_handle_updates_current_update_done(self, setup_event_cycle):
        """Test handling updates when update is scheduled for current timeslot and night is done."""
        event_cycle, comps = setup_event_cycle

        site = comps['params'].sites
        night_idx = NightIndex(0)
        current_timeslot = TimeslotIndex(5)

        update_event = MockEvent(datetime.now(), "Update Event")
        time_record = TimeCoordinateRecord(
            timeslot_idx=TimeslotIndex(5),
            event=update_event,
            done=True,
            perform_time_accounting=True
        )
        next_update = time_record

        plans = MagicMock(spec=Plans)
        nightly_timeline = MagicMock(spec=NightlyTimeline)

        # Mock the method we'll call
        event_cycle._perform_time_accounting = MagicMock()

        result_plans = event_cycle._handle_updates(
            site, night_idx, current_timeslot, next_update, plans, nightly_timeline
        )

        # Should perform time accounting but not create new plan
        event_cycle._perform_time_accounting.assert_called_once()

        # Should not change plans but mark night as done
        assert result_plans == plans

    def test_perform_time_accounting_not_done(self, setup_event_cycle):
        """Test time accounting when night is not done."""
        event_cycle, comps = setup_event_cycle

        site = comps['params'].sites
        night_idx = NightIndex(0)

        update_event = MockEvent(datetime.now(), "Update Event")
        update = TimeCoordinateRecord(
            timeslot_idx=TimeslotIndex(5),
            event=update_event,
            done=False,
            perform_time_accounting=True
        )

        end_timeslot_bounds = {site: update.timeslot_idx}
        plans = MagicMock(spec=Plans)
        current_timeslot = TimeslotIndex(5)
        nightly_timeline = MagicMock(spec=NightlyTimeline)

        event_cycle._perform_time_accounting(
            site, night_idx, update, end_timeslot_bounds, plans, current_timeslot, nightly_timeline
        )

        # Should call time_accounting but not add a final plan
        comps['scp'].collector.time_accounting.assert_called_once_with(
            plans=plans,
            sites=frozenset({site}),
            end_timeslot_bounds=end_timeslot_bounds
        )
        nightly_timeline.get_final_plan.assert_not_called()
        nightly_timeline.add.assert_not_called()


    def test_create_new_plan_unblocked(self, setup_event_cycle):
        """Test creating a new plan when site is unblocked."""
        event_cycle, comps = setup_event_cycle

        site = comps['params'].sites
        night_idx = NightIndex(0)
        current_timeslot = TimeslotIndex(5)

        update_event = MockEvent(datetime.now(), "Update Event")
        update = TimeCoordinateRecord(
            timeslot_idx=TimeslotIndex(5),
            event=update_event,
            done=False,
            perform_time_accounting=True
        )

        plans = MagicMock(spec=Plans)
        nightly_timeline = MagicMock(spec=NightlyTimeline)
        new_plans = MagicMock(spec=Plans)
        site_plan = MagicMock()
        new_plans.__getitem__.return_value = site_plan

        comps['scp'].run.return_value = new_plans

        event_cycle.change_monitor.is_site_unblocked.return_value = True

        result_plans = event_cycle._create_new_plan(
            site, night_idx, current_timeslot, update, plans, nightly_timeline
        )

        # Should run SCP and add plan to timeline
        comps['scp'].run.assert_called_once_with(site, np.array([night_idx]), current_timeslot)
        nightly_timeline.add.assert_called_once_with(
            NightIndex(night_idx), site, current_timeslot, update_event, site_plan
        )
        assert result_plans == new_plans

    def test_create_new_plan_blocked(self, setup_event_cycle):
        """Test creating a new plan when site is blocked."""
        event_cycle, comps = setup_event_cycle

        site = comps['params'].sites
        night_idx = NightIndex(0)
        current_timeslot = TimeslotIndex(1)

        update_event = MockEvent(datetime.now(), "Update Event")
        update = TimeCoordinateRecord(
            timeslot_idx=TimeslotIndex(5),
            event=update_event,
            done=False,
            perform_time_accounting=True
        )

        plans = MagicMock(spec=Plans)
        nightly_timeline = MagicMock(spec=NightlyTimeline)

        event_cycle.change_monitor.is_site_unblocked.return_value = False

        result_plans = event_cycle._create_new_plan(
            site, night_idx, current_timeslot, update, plans, nightly_timeline
        )

        # Should not run SCP but add None plan to timeline
        comps['scp'].run.assert_not_called()
        nightly_timeline.add.assert_called_once_with(
            NightIndex(night_idx), site, current_timeslot, update_event, None
        )
        assert result_plans == plans
