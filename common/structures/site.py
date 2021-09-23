from enum import Enum, unique
from astropy.coordinates import EarthLocation
import pytz


# TODO: This file will have to be changed to make this compatible with different observatories.

@unique
class Site(Enum):
    """
    The sites (telescopes) available to an observation.
    """
    GS = 'gs'
    GN = 'gn'


# Full site names.
SITE_NAMES = {Site.GS: 'gemini_south',
              Site.GN: 'gemini_north'}

# Dict from Site to geographical location.
GEOGRAPHICAL_LOCATIONS = {site: EarthLocation.of_site(name) for site, name in SITE_NAMES.items()}

# Dict from Site to timezone.
TIME_ZONES = {site: pytz.timezone(location.info.meta['timezone']) for site, location in GEOGRAPHICAL_LOCATIONS.items()}

# Site abbreviations for FPUs, gratings, etc.
SITE_ABBREVIATION = {Site.GS: 'S', Site.GN: 'N'}

# Site zip timestamps.
SITE_ZIP_EXTENSIONS = {Site.GS: "-0830.zip", Site.GN: "-0715.zip"}
