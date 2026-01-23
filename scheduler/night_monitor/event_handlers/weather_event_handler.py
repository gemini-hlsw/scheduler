# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Dict, Tuple, Callable
from lucupy.minimodel import ImageQuality, CloudCover, Site, VariantSnapshot
from scheduler.clients import SchedulerQueue
from astropy.coordinates import Angle
import astropy.units as u
from scheduler.core.events.queue import WeatherChangeEvent
from datetime import datetime, UTC

from .event_handler import EventHandler

__all__ = ['WeatherEventHandler']

class WeatherEventHandler(EventHandler):

    def _build_dispatch_map(self) -> Dict[str, Tuple[Callable, Callable]]:
        return {
            "weather_change": (
                self.parse_weather_change,
                self._on_weather_change,
            ),
        }

    def parse_weather_change(self, raw_event: dict) -> dict:
        # Get weather changes
        # Dict of type {'weatherUpdates': {'site': string, 'imageQuality': number, 'cloudCover': number, 'windDirection': number, 'windSpeed': number}}
        site = Site[raw_event['weatherUpdates']['site']]
        image_quality = ImageQuality(raw_event['weatherUpdates']['imageQuality'])
        cloud_cover = CloudCover(raw_event['weatherUpdates']['cloudCover'])
        wind_speed = raw_event['weatherUpdates']['windSpeed']
        wind_direction = raw_event['weatherUpdates']['windDirection']

        variant = VariantSnapshot(iq=image_quality,
                                  cc=cloud_cover,
                                  wind_dir=Angle(wind_direction, unit=u.deg),
                                  wind_spd=wind_speed * (u.m / u.s))
        
        return WeatherChangeEvent(variant_change=variant,
                                  time=datetime.now(UTC),
                                  site=site,
                                  description=f"weather changed in site {site.name}")
    
    async def _on_weather_change(self, event: WeatherChangeEvent, scheduler_queue: SchedulerQueue):
        await scheduler_queue.add_schedule_event(reason='Weather changed', event=event)
