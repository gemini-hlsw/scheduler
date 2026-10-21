# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import asyncio
from datetime import datetime, timedelta, UTC
from typing import FrozenSet, List
from lucupy import sky
from lucupy.minimodel import Site
from astropy.time import Time
from scheduler.core.builder.modes import SchedulerModes, app_mode
from scheduler.core.events.queue.events import (
    NightEvent,
    TwilightEvent,
    MorningTwilightEvent,
    EveningTwilightEvent,
    EndOfNightEvent,
    OnDemandScheduleEvent,
)
from scheduler.core.events.queue.scheduler_queue_client import SchedulerQueue
from scheduler.engine.params import default_operation_start
from scheduler.services.logger_factory import create_logger

_logger = create_logger(__name__)

__all__ = ["NightTracker"]


class NightTracker:
    """
    Handles ODB events. To check the different subscriptions go to ODBEventSource.

    Fields:
        CHECK_INTERVAL: ClassVar[timedelta] = timedelta(minutes=10)
    """

    CHECK_INTERVAL = 30  # seconds

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
        self.sites = sites
        self.scheduler_queue = scheduler_queue
        self.sorted_night_events: List[NightEvent] = []

    def _compute_sorted_night_events(self) -> List[NightEvent]:
        """
        Compute and sort every site's night events.
        It creates a continuous night:  GS evening twilight to GN morning twilight.
        """
        self.date = default_operation_start(datetime.now(UTC))
        all_events = []
        for site in self.sites:
            all_events.extend(self.calculate_night_events(self.date, site))

        sorted_events = sorted(all_events, key=lambda x: x.time)

        # Add end of night event
        sorted_events.append(
            EndOfNightEvent(
                description="End of Night", time=(sorted_events[-1].time + timedelta(minutes=5)), site="Both"
            ),
        )
        return sorted_events

    async def prepare(self) -> None:
        """
        Compute the night events off the event loop.
        """
        if self.sorted_night_events:
            return
        self.sorted_night_events = await asyncio.to_thread(self._compute_sorted_night_events)
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
            astropy_date, site.location, site.timezone
        )

        night_events = [
            NightEvent(description=f"Midnight at {site.name}", time=midnight[0].to_datetime(timezone=UTC), site=site),
            TwilightEvent(description=f"Sunset at {site.name}", time=sunset.to_datetime(timezone=UTC), site=site),
            TwilightEvent(description=f"Sunrise at {site.name}", time=sunrise.to_datetime(timezone=UTC), site=site),
            EveningTwilightEvent(
                description=f"Evening 12° Twilight at {site.name}", time=even_12twi.to_datetime(timezone=UTC), site=site
            ),
            MorningTwilightEvent(
                description=f"Morning 12° Twilight at {site.name}", time=morn_12twi.to_datetime(timezone=UTC), site=site
            ),
            NightEvent(description=f"Moonrise at {site.name}", time=moonrise.to_datetime(timezone=UTC), site=site),
            NightEvent(description=f"Moonset at {site.name}", time=moonset.to_datetime(timezone=UTC), site=site),
        ]

        return night_events

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
            "End of Night",
        ]

        return event.description in trigger_events

    async def _wait_and_add(self, now: datetime, event: NightEvent) -> None:
        """
        Calculates the selected event and wait to add them to the queue.

        Args:
          now (datetime): Current time to calculate when the system started.
          event (NightEvent): NightEvent to send.

        """
        wait_time = (event.time - now).total_seconds()
        if wait_time > 0:
            _logger.info(f'Event {event.description} set to trigger in {wait_time} seconds at {event.time}')
            try:
                await asyncio.sleep(wait_time)
                await self.scheduler_queue.add_schedule_event(event)
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
        # Compute the night events off the loop before tracking them.
        await self.prepare()

        schedule_queue = self.scheduler_queue
        now = datetime.now(UTC)

        if app_mode != SchedulerModes.OPERATION:
            _logger.info("Starting non-real-time tracking of night events")
            for night_event in self.sorted_night_events:
                if self._should_trigger_plan(night_event):
                    await schedule_queue.add_schedule_event(night_event)
            return

        # Real-time mode
        _logger.info("Starting real-time tracking of night events")
        # One continuous night after another for as long as the process lives. Restarts land at any
        # time, so stopping at End of Night left the next night, GN included, unplanned until the next restart.
        while True:
            twilights = [
                ne for ne in self.sorted_night_events if isinstance(ne, (EveningTwilightEvent, MorningTwilightEvent))
            ]
            if twilights and twilights[0].time <= now < twilights[-1].time:
                # Started inside the night: its past twilights are skipped below, so without this
                # nothing plans until the next twilight, which can be GN's morning one.
                await schedule_queue.add_schedule_event(
                    OnDemandScheduleEvent(site=None, description="Started during the night", time=now)
                )

            filtered_night_events = [ne for ne in self.sorted_night_events if self._should_trigger_plan(ne)]
            tasks = []
            for event in filtered_night_events:
                task = asyncio.create_task(self._wait_and_add(now, event))
                tasks.append(task)

            try:
                await asyncio.gather(*tasks, return_exceptions=True)
                _logger.info("Finished real-time tracking of night events")
            except asyncio.CancelledError:
                # Shutdown
                _logger.info('Night event tracking cancelled.')
                raise

            # Only move on once End of Night has really passed: recomputing earlier brings the same
            # night back, and the loop would spin re-queueing it.
            remaining = (self.sorted_night_events[-1].time - datetime.now(UTC)).total_seconds()
            if remaining > 0:
                await asyncio.sleep(remaining)

            # EndofNight is past GN's morning twilight, so the anchor has moved to the next night.
            self.sorted_night_events = []
            await self.prepare()
            now = datetime.now(UTC)

        # while filtered_night_events:
        #  current_time = Time(datetime.now(tz=timezone.utc), scale='utc')
        #  next_night_event = self.sorted_night_events[0]
        #  print(f'current: {current_time} and next event {next_night_event.description}: {next_night_event.time}')
        #  if current_time >= next_night_event.time:
        #    _logger.debug(f"Event Triggered: {next_night_event.description} at {current_time.iso}")
        #    self.sorted_night_events.pop(0)
        #    if self.should_trigger_plan(next_night_event):
        #      await schedule_queue.add_schedule_event(next_night_event)
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
