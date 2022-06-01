from enum import Enum

import astropy.coordinates
import pytz


class SiteInformation:
    def __init__(self,
                 name: str,
                 coordinate_center: str,
                 astropy_lookup: str = None):
        """
        AstroPy location lookups for Gemini North and South are of the form:
        * gemini_north
        * gemini_south
        This conversion will happen automatically if astropy_lookup is None.

        If necessary, other observatories should provide hard astropy_lookup values.

        The following is also included here:
        * name: the name of the site in human-readable format
        * timezone: time zone information
        * location: the AstroPy location lookup (astropy.coordinates.earth) of the site
        * coordinate_center: coordinate center for Ephemeris lookups
        """
        if astropy_lookup is None:
            astropy_lookup = name.lower().replace(' ', '_')

        self.name = name
        self.coordinate_center = coordinate_center

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


class Site(Enum):
    """
    The sites belonging to the observatory using the Scheduler.

    This will have to be customized by a given observatory if used independently
    of Gemini.
    """
    GN = SiteInformation('Gemini North', '568@399')
    GS = SiteInformation('Gemini South', 'I11@399')


ALL_SITES = frozenset(s for s in Site)
