from enum import Enum
from typing import Optional

import astropy.coordinates
import pytz


class Site(Enum):
    """
    The sites belonging to the observatory using the Scheduler.

    This will have to be customized by a given observatory if used independently
    of Gemini.
    """
    GN = ('Gemini North', '568@399')
    GS = ('Gemini South', 'I11@399')

    def __init__(self, site_name: str, coordinate_center: str, astropy_lookup: Optional[str] = None):
        self.site_name = site_name
        self.coordinate_center = coordinate_center

        if astropy_lookup is None:
            astropy_lookup = site_name.lower().replace(' ', '_')

        try:
            self.location = astropy.coordinates.EarthLocation.of_site(astropy_lookup)
        except astropy.coordinates.UnknownSiteException as e:
            msg = f'Unknown site lookup: {astropy_lookup}.'
            raise ValueError(e, msg)

        timezone_info = self.location.info.meta['timezone']
        try:
            self.timezone = pytz.timezone(timezone_info)
        except pytz.UnknownTimeZoneError as e:
            msg = f'Unknown time zone lookup: {timezone_info}.'
            raise ValueError(e, msg)


ALL_SITES = frozenset(s for s in Site)
