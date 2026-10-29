# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import replace
from typing import Dict, Tuple, Callable
from lucupy.minimodel import ImageQuality, CloudCover, Site, VariantSnapshot, WaterVapor
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

    def parse_weather_change(self, raw_event: dict) -> WeatherChangeEvent:
        # Get weather changes
        # Dict of type {'weatherUpdates': {'site': string, 'imageQuality': number, 'cloudCover': number, 'waterVapor': number, 'windDirection': number, 'windSpeed': number}}
        site = Site[raw_event['weatherUpdates']['site']]
        image_quality = ImageQuality(raw_event['weatherUpdates']['imageQuality'])
        cloud_cover = CloudCover(raw_event['weatherUpdates']['cloudCover'])
        # The weather service may not report water vapor: assume the worst (WV ANY).
        raw_water_vapor = raw_event['weatherUpdates'].get('waterVapor')
        water_vapor = WaterVapor(1.0 if raw_water_vapor is None else raw_water_vapor)
        wind_speed = raw_event['weatherUpdates']['windSpeed']
        wind_direction = raw_event['weatherUpdates']['windDirection']

        variant = VariantSnapshot(
            iq=image_quality,
            cc=cloud_cover,
            wv=water_vapor,
            wind_dir=Angle(wind_direction, unit=u.deg),
            wind_spd=wind_speed * (u.m / u.s),
        )

        # Placeholder: the parser is synchronous, so _on_weather_change restamps this with the
        # time the plan is running on before the event reaches the Engine.
        return WeatherChangeEvent(
            variant_change=variant,
            time=datetime.now(UTC),
            site=site,
            description=self._describe(site, variant, water_vapor_reported=raw_water_vapor is not None),
        )

    @staticmethod
    def _describe(site: Site, variant: VariantSnapshot, water_vapor_reported: bool) -> str:
        """E.g. 'Weather changed at GN: IQ70, CC50, WV80, wind 5.0 m/s from 90°', for the UI and the logs."""
        wv = f"{variant.wv}" if water_vapor_reported else f"{variant.wv} (not reported)"
        return (
            f"Weather changed at {site.name}: {variant.iq}, {variant.cc}, {wv}, "
            f"wind {variant.wind_spd.value:.1f} m/s from {variant.wind_dir.deg:.0f}°"
        )

    async def _on_weather_change(self, event: WeatherChangeEvent):
        when = await self._reference_time(event.site)
        await self.scheduler_queue.add_schedule_event(replace(event, time=when))
