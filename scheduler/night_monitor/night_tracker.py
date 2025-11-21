# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from lucupy import sky
from astropy.time import Time
from astropy import units as u
from time import sleep
from scheduler.clients.scheduler_queue_client import schedule_queue
from scheduler.core.builder.modes import SchedulerModes, app_mode
import datetime

__all__ = ["NightTracker"]

class NightTracker:
  """
    Handles ODB events. To check the different subscriptions go to ODBEventSource.

    Fields:
        CHECK_INTERVAL: ClassVar[timedelta] = timedelta(minutes=10)
    """
  
  CHECK_INTERVAL = 30 # seconds

  def __init__(self, date: datetime.datetime, sites: list):
    """
    Constructor for NightTracker.
    Args:
        date (datetime.datetime): The date for which to track night events.
        sites (list): List of site objects to track events for.
    """
    # Set night date
    self.date = date

    # Precompute night events for each site as an array of tuples
    all_events = []
    for site in sites:
      (midnight, sunset, sunrise, even_12twi, morn_12twi, moonrise, moonset) = sky.night_events(Time(date, format='datetime'), site.location, site.timezone)
      all_events.extend([
        (midnight[0], f"Midnight at {site.name}"),
        (sunset, f"Sunset at {site.name}"),
        (sunrise, f"Sunrise at {site.name}"),
        (even_12twi, f"Evening 12° Twilight at {site.name}"),
        (morn_12twi, f"Morning 12° Twilight at {site.name}"),
        (moonrise, f"Moonrise at {site.name}"),
        (moonset, f"Moonset at {site.name}"),
      ])
    
    # Sort events by time
    self.sorted_night_events = sorted(all_events, key=lambda x: x[0])

    # Add end of night event
    self.sorted_night_events.append((self.sorted_night_events[-1][0] + 5 * u.min, "End of Night"))

  def start_tracking(self):
    """
    This should be a thread or async process that
    In RT should add events to the scheduler queue when an event time is reached
    In non-RT should add all events to the scheduler queue at once
    """
    if app_mode != SchedulerModes.OPERATION:
      for event_time, event_desc in self.sorted_night_events:
        # TODO: Add proper event object to the queue
        schedule_queue.add_schedule_event()
      return

    # Real-time mode
    while len(self.sorted_night_events) > 0:
      current_time = Time.now()
      next_event_time, event_description = self.sorted_night_events[0]
      if current_time >= next_event_time:
        print(f"Event Triggered: {event_description} at {current_time.iso}")
        self.sorted_night_events.pop(0)
        # TODO: Add proper event object to the queue
        schedule_queue.add_schedule_event()
      else:
        sleep(self.CHECK_INTERVAL)  # Sleep for a while before checking again

  def __str__(self):
    """
    Returns a string representation of the NightTracker object.
    """
    text = f"Night Events for {self.date.date()}:"
    for event in self.sorted_night_events:
      text += f"\n - {event[0]} - {event[1]}"
    return text


if __name__ == "__main__":
  from lucupy.minimodel import Site

  # Example usage
  date = datetime.datetime(2024, 10, 1)
  sites = [Site.GS, Site.GN]
  tracker = NightTracker(date, sites)

  # Display the night events for all sites
  print(tracker)

  # Start tracking
  # This should be a thread or async process
  # tracker.start_tracking()