from enum import Enum, unique
from astropy.coordinates import EarthLocation


# TODO: This file will have to be changed to make this compatible with different observatories.

@unique
class Site(Enum):
    """
    The sites (telescopes) available to an observation.
    """
    GS = 'gs'
    GN = 'gn'


# Dict from Site to geographical location.
GEOGRAPHICAL_LOCATIONS = {Site.GS: EarthLocation.of_site('gemini_south'),
                          Site.GN: EarthLocation.of_site('gemini_north')}

# Site abbreviations for FPUs, gratings, etc.
SITE_ABBREVIATION = {Site.GS: 'S', Site.GN: 'N'}

# Site zip timestamps.
SITE_ZIPS = {Site.GS: "-0830.zip", Site.GN: "-0715.zip"}
