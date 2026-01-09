# Copyright (c) 2016-2025 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from typing import Dict, Tuple, Callable

from .event_handler import EventHandler

__all__ = ['WeatherEventHandler']

class WeatherEventHandler(EventHandler):

    def _build_dispatch_map(self) -> Dict[str, Tuple[Callable, Callable]]:
        return {}
