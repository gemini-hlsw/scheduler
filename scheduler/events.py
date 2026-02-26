from dataclasses import dataclass
from datetime import datetime, timedelta

from astropy.time import Time
from lucupy.minimodel import Site, TimeslotIndex
from lucupy.timeutils import time2slots


@dataclass
class NightEvent:
  description: str
  time: Time
  site: Site


@dataclass
class OnDemandScheduleEvent:
  description: str
  time: datetime



def to_timeslot_idx(
  time: datetime,
  twi_eve_time: datetime,
  time_slot_length: timedelta
) -> TimeslotIndex:
  """Given an event, calculate the timeslot offset it falls into relative to another datetime.
  This would typically be the twilight of the night on which the event occurs, hence the name twi_eve_time.

  Args:
      time (datetime): time stamp of the event.
      twi_eve_time (datetime): Evening twilight time.
      time_slot_length (timedelta): Set time slot length in which the Scheduler work.

  Returns:
      TimeslotIndex: The timeslot offset relative to the twilight.
  """
  # print(time, twi_eve_time, time_slot_length)
  time_from_twilight = time - twi_eve_time

  time_slots_from_twilight = time2slots(time_slot_length, time_from_twilight)
  return TimeslotIndex(time_slots_from_twilight)