from dataclasses import dataclass
from datetime import datetime, timedelta
import dateutil.parser
from common.minimodel import Target, TargetTag, Site
import numpy as np
import numpy.typing as npt
import requests
from typing import Tuple

import os

@dataclass
class EphemerisCoordinates:
    """
    Both ra and dec are in radians.

    """
    ra: npt.NDArray[float]
    dec: npt.NDArray[float]
    time: npt.NDArray[float]

    @staticmethod
    def dms2deg(dms):

        if dms is None:
            return None

        d = float(dms[0])
        m = float(dms[1])
        s = float(dms[2])
        sign = dms[3]
        dd = d + m / 60. + s / 3600.

        if sign == '-':
            dd *= -1.

        return dd

    @staticmethod
    def dms2rad(dms):
        if dms is None:
            return None
        
        dd = EphemerisCoordinates.dms2deg(dms)
        rad = dd * np.pi / 180.
        return rad

    @staticmethod
    def hms2rad(hms):
        if hms is None:
            return None

        h = float(hms[0])
        m = float(hms[1])
        s = float(hms[2])
        hours = h + m / 60. + s / 3600.
        rad = hours * np.pi / 12.

        return rad


class HorizonsClient:
    """
    API to interact with the Horizons service
    """

    # A not-complete list of solar system major body Horizons IDs
    bodies = {'mercury': '199', 'venus': '299', 'mars': '499', 'jupiter': '599', 'saturn': '699',
               'uranus': '799', 'neptune': '899', 'pluto': '999', 'io': '501'}

    def __init__(self,
                 site: Site,
                 path: str = 'horizons/data',
                 start: datetime = None,
                 end: datetime = None):
        self.path = path
        self.url = 'https://ssd.jpl.nasa.gov/horizons_batch.cgi'
        self.start = start
        self.end = end
        self.site = site
     
    
    @staticmethod
    def generate_horizons_id(designation: str) -> str:
        des = designation.lower()
        return HorizonsClient.bodies[des] if des in HorizonsClient.bodies else designation
    
    @staticmethod
    def _angular_distance(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
        """
        Calculate the angular distance between two points on the sky.
        based on https://github.com/gemini-hlsw/lucuma-core/blob/master/modules/core/shared/src/main/scala/lucuma/core/math/Coordinates.scala#L52
        """
        φ1 = dec1
        φ2 = dec2
        delta_φ = dec2 - dec1
        delta_λ = ra2 - ra1
        a = np.sin(delta_φ / 2)**2 + np.cos(φ1) * np.cos(φ2) * np.sin(delta_λ / 2)**2
        return 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @staticmethod
    def interpolate(ra1: float, dec1: float, ra2: float, dec2: float, f: float) -> float:
        """
        Interpolate between two coordinates.
        based on https://github.com/gemini-hlsw/lucuma-core/blob/master/modules/core/shared/src/main/scala/lucuma/core/math/Coordinates.scala#L63-L89
        """
        # calculate angular distance to radians
        delta = HorizonsClient.angular_distance(ra1, dec1, ra2, dec2)
        if delta == 0:
            return ra1, dec1
        else:

            φ1 = dec1
            φ2 = dec2
            λ1 = ra1
            λ2 = ra2
            a = np.sin((1 - f) * delta) / np.sin(delta)
            b = np.sin(f * delta) / np.sin(delta)
            x = a * np.cos(φ1) * np.cos(λ1) + b * np.cos(φ2) * np.cos(λ2)
            y = a * np.cos(φ1) * np.sin(λ1) + b * np.cos(φ2) * np.sin(λ2)
            z = a * np.sin(φ1) + b * np.sin(φ2)
            φi = np.arctan2(z, np.sqrt(x * x + y * y))
            λi = np.arctan2(y, x)
            return λi, φi # there is a transformation here

    def _form_horizons_name(self, tag: TargetTag, designation: str) -> str:
        """
        Formats the name of the body
        """
        if tag is TargetTag.COMET:
            name = f'DES={designation};CAP'
        elif tag is TargetTag.ASTEROID:
            name = f'DES={designation};'
        else:
            name = self.generate_horizons_id(designation)
        return name

    def _time_bounds(self) -> Tuple[str, str]:
        """
        Returns the start and end times based on the given date
        """
        return self.start.strftime('%Y%m%d_%H%M'), self.end.strftime('%Y%m%d_%H%M')

    def _get_ephemeris_file(self, name: str) -> str:
        """
        Returns the ephemeris file name
        """
        start, end = self._time_bounds()
        return os.path.join(self.path, f"{self.site.name}_{name.replace(' ', '').replace('/', '')}_{start}-{end}.eph")
    
    def query(self,
              target: str,
              step: str = '1m',
              make_ephem: str = 'YES',
              cal_format: str = 'CAL',
              quantities: str = '1',
              object_data: str = 'NO',
              daytime: bool = False,
              csvformat: str = 'NO',
              airmass: int = 3) -> requests.Response:
     
        # The items and order follow the JPL/Horizons batch example:
        # ftp://ssd.jpl.nasa.gov/pub/ssd/horizons_batch_example.long
        # and
        # ftp://ssd.jpl.nasa.gov/pub/ssd/horizons-batch-interface.txt
        # Note that spaces should be converted to '%20'
        
        skip_day = 'NO' if daytime else 'YES'
        center = '568@399' if self.site is Site.GN else 'I11@399'

        params = {'batch':1}
        params['COMMAND']    = "'" + target + "'"
        params['OBJ_DATA']   = object_data # Toggles return of object summary data (YES or NO)
        params['MAKE_EPHEM'] = make_ephem # Toggles generation of ephemeris (YES or NO)
        params['TABLE_TYPE'] = 'OBSERVER'      # OBSERVER, ELEMENTS, VECTORS, or APPROACH 
        params['CENTER']     = center     # Set coordinate origin. MK=568, CP=I11, Earth=399
        params['REF_PLANE']  = None            # Table reference plane (ECLIPTIC, FRAME or BODY EQUATOR)
        params['COORD_TYPE'] = None            # Type of user coordinates in SITE_COORD
        params['SITE_COORD'] = None            # '0,0,0'
        
        params['START_TIME'] = self.start      # Ephemeris start time YYYY-MMM-DD {HH:MM} {UT/TT}
        params['STOP_TIME']  = self.end        # Ephemeris stop time YYYY-MMM-DD {HH:MM}
        params['STEP_SIZE']  = "'" + step + "'" # Ephemeris step: integer# {units} {mode}
        params['TLIST']      = None            # Ephemeris time list

        # This only works for small numbers (~<1000) of times:
        #tlist = ' '.join(map(str,numpy.arange(2457419.5, 2457600.0, 0.3)))
        #params['TLIST']      = tlist            # Ephemeris time list

        params['QUANTITIES'] = quantities # Desired output quantity codes
        params['REF_SYSTEM'] = 'J2000'         # Reference frame
        params['OUT_UNITS']  = None            # VEC: Output units
        params['VECT_TABLE'] = None            # VEC: Table format
        params['VECT_CORR']  = None            # VEC: correction level
        params['CAL_FORMAT'] = cal_format # OBS: Type of date output (CAL, JD, BOTH)
        params['ANG_FORMAT'] = 'HMS'           # OBS: Angle format (HMS or DEG)
        params['APPARENT']   = None            # OBS: Apparent coord refract corr (AIRLESS or REFRACTED)
        params['TIME_DIGITS'] = 'MINUTES'      # OBS: Precision (MINUTES, SECONDS, or FRACSEC)
        params['TIME_ZONE']  = None            # Local civil time offset relative to UT ('+00:00')
        params['RANGE_UNITS'] = None           # OBS: range units (AU or KM)
        params['SUPPRESS_RANGE_RATE'] = 'NO'   # OBS: turn off output of delta-dot and rdot
        params['ELEV_CUT']   = '-90'           # OBS: skip output when below elevation
        params['SKIP_DAYLT'] = skip_day   # OBS: skip output when daylight
        params['SOLAR_ELONG'] = "'0,180'"      # OBS: skip output outside range
        params['AIRMASS']    = airmass    # OBS: skip output when airmass is > cutoff
        params['LHA_CUTOFF'] = None            # OBS: skip output when hour angle is > cutoff
        params['EXTRA_PREC'] = 'YES'           # OBS: show additional output digits (YES or NO)
        params['CSV_FORMAT'] = csvformat  # Output in comma-separated value format (YES or NO)
        params['VEC_LABELS'] = None            # label each vector component (YES or NO)
        params['ELM_LABELS'] = None            # label each osculating element
        params['TP_TYPE']    = None            # Time of periapsis for osculating element tables
        params['R_T_S_ONLY'] = 'NO'            # Print only rise/transit/set (NO, TVH, GEO, RAD, YES)
        # Skiping the section of close-approch parameters...
        # Skiping the section of heliocentric ecliptic osculating elements...
        response = requests.get(self.url, params=params)

        return response

    def get_ephemerides(self, target: Target, overwrite: bool = False) -> EphemerisCoordinates:
 
        horizons_name = self._form_horizons_name(target.tag, target.des)
        file = self._get_ephemeris_file(horizons_name)

        if not overwrite and os.path.exists(file):
            with open(file, 'r') as f:
                lines = f.readlines()
        
        else:
            res = self.query(horizons_name)
            lines = res.text.splitlines()
            if file != None:
                with open(file, 'w') as f:
                    f.write(res.text)

        firstline = lines.index('$$SOE') + 1
        lastline = lines.index('$$EOE') - 1
        time = np.array([])
        ra = np.array([])
        dec = np.array([])

        for line in lines[firstline:lastline]:
            if line and line[7:15] != 'Daylight' and line[7:14] != 'Airmass':

                values = line.split(' ')
                values = d.split(' ')
                rah = int(values[-6])
                ram = int(values[-5])
                ras = float(values[-4])
                decg = values[-3][0] # sign
                decd = int(values[-3][1:3])
                decm = int(values[-2])
                decs = float(values[-1])

                time = np.append(time, dateutil.parser.parse(line[1:18]))
                ra = np.append(ra, EphemerisCoordinates.hms2rad([rah, ram, ras]))
                dec = np.append(dec, EphemerisCoordinates.dms2rad([decd, decm, decs, decg]))
        
        return EphemerisCoordinates(ra, dec, time)

    @staticmethod
    def _bracket(times: np.array, time: datetime) -> Tuple[datetime, datetime]:
        """
        Return both lower and upper of the given time: i.e., the closest elements on either side. 
        """
        return times[times > time].min(), times[times < time].max()

    @staticmethod
    def interpolate_ephemeris(ephemeris: EphemerisCoordinates, time: datetime) -> Tuple[float, float]:
        """
        Interpolate ephemeris to a given time.
        """
        a, b = HorizonsClient._bracket(ephemeris.time, time)
        # Find indexes for each bound
        i_a, i_b = np.where(ephemeris.time == a)[0][0], np.where(ephemeris.time == b)[0][0]
        factor = (time - a / b - a).timestamp() * 1000
        ra, dec = HorizonsClient.interpolate(ephemeris.ra[i_a], 
                                             ephemeris.dec[i_a], 
                                             ephemeris.ra[i_b], 
                                             ephemeris.dec[i_b], 
                                             factor)
        return ra, dec
