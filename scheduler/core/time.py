from datetime import datetime, timedelta
from math import ceil

from lucupy.minimodel import NightIndex


class TimeHandler:

    def __init__(self, num_nights_to_schedule, start, time_slot_length):
        self.start = start
        self.num_nights_to_schedule = num_nights_to_schedule
        self.time_slot_length = time_slot_length

    def dt2night_index(self, dt: datetime) -> NightIndex:
        days = dt.day - self.start.day
        if days > self.num_nights_to_schedule:
            raise ValueError('Date outside of the number of night to schedule!')
        return NightIndex(days)

    def time2slots(self, time: timedelta) -> int:
        return ceil(time / self.time_slot_length)
