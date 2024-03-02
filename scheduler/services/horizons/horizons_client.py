# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import contextlib
import os
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Mapping, Tuple, final

import dateutil.parser
import numpy as np
import requests
from lucupy.helpers import dms2rad, hms2rad
from lucupy.minimodel import NonsiderealTarget, TargetTag, Site

from .coordinates import Coordinates
from .ephemeris_coordinates import EphemerisCoordinates
from scheduler.services import logger_factory

__all__ = [
    'HorizonsClient',
    'horizons_session',
]


logger = logger_factory.create_logger(__name__)


@final
@dataclass(frozen=True)
class HorizonsClient:
    """
    API to interact with the Horizons service
    """
    site: Site
    path: str = os.path.join('scheduler', 'services', 'horizons', 'data')
    airmass: int = 3
    start: datetime = None
    end: datetime = None

    url: ClassVar[str] = 'https://ssd.jpl.nasa.gov/horizons_batch.cgi'
    # A not-complete list of solar system major body Horizons IDs
    bodies: ClassVar[Mapping[str, str]] = {'mercury': '199', 'venus': '299', 'mars': '499', 'jupiter': '599',
                                           'saturn': '699', 'uranus': '799', 'neptune': '899', 'pluto': '999',
                                           'io': '501'}

    FILE_DATE_FORMAT: ClassVar[str] = '%Y%m%d_%H%M'

    @staticmethod
    def generate_horizons_id(designation: str) -> str:
        des = designation.lower()
        return HorizonsClient.bodies[des] if des in HorizonsClient.bodies else designation

    def _time_bounds(self) -> Tuple[str, str]:
        """
        Returns the start and end times based on the given date
        """
        return self.start.strftime(HorizonsClient.FILE_DATE_FORMAT), self.end.strftime(HorizonsClient.FILE_DATE_FORMAT)

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
              csvformat: str = 'NO') -> requests.Response:

        # The items and order follow the JPL/Horizons batch example:
        # ftp://ssd.jpl.nasa.gov/pub/ssd/horizons_batch_example.long
        # and
        # ftp://ssd.jpl.nasa.gov/pub/ssd/horizons-batch-interface.txt
        # Note that spaces should be converted to '%20'

        skip_day = 'NO' if daytime else 'YES'

        center = self.site.coordinate_center

        params = {'batch': 1,
                  'COMMAND': "'" + target + "'",
                  'OBJ_DATA': object_data,
                  'MAKE_EPHEM': make_ephem,
                  'TABLE_TYPE': 'OBSERVER',
                  'CENTER': center,
                  'REF_PLANE': None,
                  'COORD_TYPE': None,
                  'SITE_COORD': None,
                  'START_TIME': self.start.strftime("'%Y-%b-%d %H:%M'"),
                  'STOP_TIME': self.end.strftime("'%Y-%b-%d %H:%M'"),
                  'STEP_SIZE': "'" + step + "'",
                  'TLIST': None,
                  'QUANTITIES': quantities,
                  'REF_SYSTEM': 'J2000',
                  'OUT_UNITS': None,
                  'VECT_TABLE': None,
                  'VECT_CORR': None,
                  'CAL_FORMAT': cal_format,
                  'ANG_FORMAT': 'HMS',
                  'APPARENT': None,
                  'TIME_DIGITS': 'MINUTES',
                  'TIME_ZONE': None,
                  'RANGE_UNITS': None,
                  'SUPPRESS_RANGE_RATE': 'NO',
                  'ELEV_CUT': '-90',
                  'SKIP_DAYLT': skip_day,
                  'SOLAR_ELONG': "'0,180'",
                  'AIRMASS': self.airmass,
                  'LHA_CUTOFF': None,
                  'EXTRA_PREC': 'YES',
                  'CSV_FORMAT': csvformat,
                  'VEC_LABELS': None,
                  'ELM_LABELS': None,
                  'TP_TYPE': None,
                  'R_T_S_ONLY': 'NO'}

        # Skipping the section of close-approach parameters.
        # Skipping the section of heliocentric ecliptic osculating elements.
        return requests.get(self.url, params=params)

    def get_ephemerides(self,
                        target: NonsiderealTarget,
                        overwrite: bool = False) -> EphemerisCoordinates:

        horizons_name = self._form_horizons_name(target.tag, target.des)
        logger.info(f'{target.des}')

        if target.tag is not TargetTag.MAJOR_BODY:
            file = self._get_ephemeris_file(target.des)
        else:
            file = self._get_ephemeris_file(horizons_name)

        if not overwrite and os.path.exists(file):
            logger.info(f'Saving ephemerides file for {target.des}')
            with open(file, 'r') as f:
                lines = list(map(lambda x: x.strip('\n'), f.readlines()))
        else:
            logger.info(f'Querying JPL/Horizons for {horizons_name}')
            res = self.query(horizons_name)
            lines = res.text.splitlines()
            if file is not None:
                with open(file, 'w') as f:
                    f.write(res.text)

        time = []
        coords = []

        try:
            firstline = lines.index('$$SOE') + 1
            lastline = lines.index('$$EOE') - 1

            for line in lines[firstline:lastline]:
                if line and line[7:15] != 'Daylight' and line[7:14] != 'Airmass':
                    values = line.split(' ')
                    rah = int(values[-6])
                    ram = int(values[-5])
                    ras = float(values[-4])
                    decg = values[-3][0]  # sign
                    decd = int(values[-3][1:3])
                    decm = int(values[-2])
                    decs = float(values[-1])

                    time.append(dateutil.parser.parse(line[1:18]))
                    coords.append(Coordinates(hms2rad(rah, ram, ras), dms2rad(decd, decm, decs, decg)))

        except ValueError as e:
            logger.error(f'Error parsing ephemerides file for {target.des}.')
            raise e

        return EphemerisCoordinates(coords, np.array(time))


@contextlib.contextmanager
def horizons_session(site, start, end, airmass) -> HorizonsClient:
    client = HorizonsClient(site, start=start, end=end, airmass=airmass)
    try:
        yield client
    finally:
        del client
    return
