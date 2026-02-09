# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
import zoneinfo
from datetime import datetime, timedelta, timezone, UTC
from typing import FrozenSet, List

from hypothesis import event
from lucupy import sky
from lucupy.minimodel import Site
from astropy.time import Time
from astropy import units as u


from scheduler.clients.scheduler_queue_client import SchedulerQueue
from scheduler.core.builder.modes import SchedulerModes, app_mode
from scheduler.events import NightEvent
from scheduler.services.logger_factory import create_logger
_logger = create_logger(__name__)

__all__ = ["NightTracker"]


class NightTracker:
  """
    Handles ODB events. To check the different subscriptions go to ODBEventSource.

    Fields:
        CHECK_INTERVAL: ClassVar[timedelta] = timedelta(minutes=10)
    """
  
  CHECK_INTERVAL = 30 # seconds

  def __init__(self, date: datetime, sites: FrozenSet[Site], scheduler_queue: SchedulerQueue):
    """
    Constructor for NightTracker.
    Args:
        date (datetime): The date for which to track night events.
        sites (Frozenset[site): List of site objects to track events for.
    """
    # Set night date
    # self.date = date
    self.date = datetime.now(UTC)
    self.scheduler_queue = scheduler_queue

    # Precompute night events for each site as an array of tuples
    all_events = []


    for site in sites:
      night_events = self.calculate_night_events(self.date, site)
      correct_night_events = self._get_correct_events(self.date, site, night_events)
      all_events.extend(correct_night_events)

    print(f'For {self.date} this are the night events:')
    for event in all_events:
      print(f'\t{event.description}: {event.time.value}')
    
    # Sort events by time
    self.sorted_night_events = sorted(all_events, key=lambda x: x.time)

    # Add end of night event
    self.sorted_night_events.append(
      NightEvent(description="End of Night", time=self.sorted_night_events[-1].time + 5 * u.min, site="Both"),
    )

    # Debugging output
    _logger.debug(self)

  @staticmethod
  def calculate_night_events(date: datetime, site: Site) -> List[NightEvent]:
    """
    Calculates for a site all the night events in the initialization date.

    Args:
      date (datetime): The date for which to track night events.
      site (Site): The site to track events for.

    Returns:
      List[NightEvent]: List of NightEvents corresponding to the site. From sunset to sunrise
    """

    astropy_date = Time(date)

    (midnight, sunset, sunrise, even_12twi, morn_12twi, moonrise, moonset) = sky.night_events(
      astropy_date,
      site.location,
      site.timezone
    )

    night_events = [
      NightEvent(description=f"Midnight at {site.name}", time=midnight[0], site=site),
      NightEvent(description=f"Sunset at {site.name}", time=sunset, site=site),
      NightEvent(description=f"Sunrise at {site.name}", time=sunrise, site=site),
      NightEvent(description=f"Evening 12° Twilight at {site.name}", time=even_12twi, site=site),
      NightEvent(description=f"Morning 12° Twilight at {site.name}", time=morn_12twi, site=site),
      NightEvent(description=f"Moonrise at {site.name}", time=moonrise, site=site),
      NightEvent(description=f"Moonset at {site.name}", time=moonset, site=site),
    ]

    return night_events

  def _get_correct_events(self, now: datetime, site: Site, events: List[NightEvent]) -> List[NightEvent]:
    """
    The system needs to verify that the night events for the initial startup time are correct.
    To keep both sites (GN and GS) running smoothly as one continuous night, the event order depends on
    the UTC startup time:
      - Early Morning (UTC): Process Gemini North (GN) twilights first, then Gemini South.
      - Other Times: Process Gemini South (GS) twilights first, then Gemini North (next day).

    Args:
      now (datetime): Current time to calculate when the system started.
      site (Site): The site to track events for.
      events (List[NightEvent]): List of NightEvents.

    Returns:
      List[NightEvent]: Original NightEvent list or the night events for the next day.

    """
    evening_twilight = events[3].time.to_datetime(timezone=UTC)
    morning_twilight = events[4].time.to_datetime(timezone=UTC)

    # If both twilights are in the past, we need the next night
    if morning_twilight < now:
      _logger.info(f"{site.name}: All events in past, getting next night")
      next_date = now + timedelta(days=1)
      new_events = self.calculate_night_events(next_date, site)
      return new_events
    # If evening twilight is in past but morning is future, we're in the middle of night
    elif evening_twilight < now < morning_twilight:
      _logger.info(f"{site.name}: Currently in astronomical night")
      return events

    # If evening twilight is in the future, this is the correct night
    if evening_twilight > now:
      _logger.info(f"{site.name}: Events are for upcoming night")
      return events

    return events

  @staticmethod
  def _should_trigger_plan(event: NightEvent) -> bool:
    """
    Determines if a given event should trigger a new plan.

    Args:
        event (NightEvent): A tuple containing the event time and description.
    Returns:
        bool: True if the event should trigger a new plan, False otherwise.
    """
    trigger_events = [
      "Evening 12° Twilight at GS",
      "Morning 12° Twilight at GS",
      "Evening 12° Twilight at GN",
      "Morning 12° Twilight at GN",
      "End of Night"
    ]

    return event.description in trigger_events

  async def _wait_and_add(self, now: datetime, event: NightEvent) -> None:
    """
    Calculates the selected event and wait to add them to the queue.

    Args:
      now (datetime): Current time to calculate when the system started.
      event (NightEvent): NightEvent to send.

    """
    wait_time = (event.time.to_datetime(timezone=UTC) - now).total_seconds()
    if wait_time > 0:
      _logger.info(f'Event {event.description} set to trigger in {wait_time} seconds at {event.time.value}')
      try:
        await asyncio.sleep(wait_time)
        await self.scheduler_queue.add_schedule_event(
          reason=f'Night event {event.description}',
          event=event
        )
      except asyncio.CancelledError:
        _logger.warning(f'Night event {event.description} wasn\'t triggered and is not in the queue.')

    else:
      _logger.info(f"Event {event.description} is in the past, skipping")


  async def start_tracking(self):
    """
    This should be a thread or async process that
    In RT should add events to the scheduler queue when an event time is reached
    In non-RT should add all events to the scheduler queue at once
    """
    schedule_queue = self.scheduler_queue
    now = datetime.now(UTC)

    if app_mode != SchedulerModes.OPERATION:
      _logger.info("Starting non-real-time tracking of night events")
      for night_event in self.sorted_night_events:
        if self._should_trigger_plan(night_event):
          await schedule_queue.add_schedule_event(
            reason=f'Night event {night_event.description}',
            event=night_event
        )
      return

    filtered_night_events = [ne for ne in self.sorted_night_events if self._should_trigger_plan(ne)]

    # Real-time mode
    _logger.info("Starting real-time tracking of night events")
    tasks = []
    for event in filtered_night_events:
      task = asyncio.create_task(self._wait_and_add(now, event))
      tasks.append(task)

    try:
      await asyncio.gather(*tasks, return_exceptions=True)
      _logger.info("Finished real-time tracking of night events")
    except asyncio.CancelledError as e:
      _logger.warning(f'Problem tracking night events: {e}')

    #while filtered_night_events:
    #  current_time = Time(datetime.now(tz=timezone.utc), scale='utc')
    #  next_night_event = self.sorted_night_events[0]
    #  print(f'current: {current_time} and next event {next_night_event.description}: {next_night_event.time}')
    #  if current_time >= next_night_event.time:
    #    _logger.debug(f"Event Triggered: {next_night_event.description} at {current_time.iso}")
    #    self.sorted_night_events.pop(0)
    #    if self.should_trigger_plan(next_night_event):
    #      await schedule_queue.add_schedule_event(
    #        reason=f'Night event {next_night_event.description}',
    #        event=next_night_event
    #      )
    #  else:
    #    await asyncio.sleep(self.CHECK_INTERVAL)  # Sleep for a while before checking again

  def __str__(self):
    """
    Returns a string representation of the NightTracker object.
    """
    text = f"Night Events for {self.date.date()}:"
    for event in self.sorted_night_events:
      text += f"\n - {event.description} - {event.time}"
    return text
