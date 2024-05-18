# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import contextlib
import dateutil.parser
import requests
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import final, ContextManager, Dict, Final

import numpy as np
from lucupy.helpers import dms2rad, hms2rad
from lucupy.minimodel import NonsiderealTarget, TargetTag, Site

from .coordinates import Coordinates
from .ephemeris_coordinates import EphemerisCoordinates
from definitions import ROOT_DIR
from scheduler.services import logger_factory

__all__ = [
    'HorizonsClient',
    'horizons_session',
]


logger = logger_factory.create_logger(__name__)


_MAJORBODY_DICT: Final[Dict[str, str]] = {
    'mercury': '199',
    'venus': '299',
    'mars': '499',
    'jupiter': '599',
    'saturn': '699',
    'uranus': '799',
    'neptune': '899',
    'pluto': '999',
    'io': '501'
}


@final
@dataclass(frozen=True)
class HorizonsClient:
    site: Site
    start: datetime
    end: datetime

    # We look up across the whole night, so the labels are simply night labels.
    date_format: str = field(default='%Y%m%d')
    path: Path = field(default=Path(ROOT_DIR) / 'scheduler' / 'services' / 'horizons' / 'data')
    url: str = field(default='https://ssd.jpl.nasa.gov/horizons_batch.cgi')

    @staticmethod
    def generate_horizons_id(designation: str) -> str:
        des = designation.lower()
        return _MAJORBODY_DICT.get(des, des)

    def _query(self,
               target: str,
               step: str = '1m',
               make_ephem: str = 'YES',
               cal_format: str = 'CAL',
               quantities: str = '1',
               object_data: str = 'NO',
               daytime: bool = False,
               csv_format: str = 'NO') -> requests.Response:

        skip_day = 'NO' if daytime else 'YES'
        center = self.site.coordinate_center

        params = {
            'batch': 1,
            'COMMAND': f"'{target}'",
            'OBJ_DATA': object_data,
            'MAKE_EPHEM': make_ephem,
            'TABLE_TYPE': 'OBSERVER',
            'CENTER': center,
            'REF_PLANE': None,
            'COORD_TYPE': None,
            'SITE_COORD': None,
            'START_TIME': self.start.strftime("'%Y-%b-%d %H:%M'"),
            'STOP_TIME': self.end.strftime("'%Y-%b-%d %H:%M'"),
            'STEP_SIZE': f"'{step}'",
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
            'AIRMASS': 100,
            'LHA_CUTOFF': None,
            'EXTRA_PREC': 'YES',
            'CSV_FORMAT': csv_format,
            'VEC_LABELS': None,
            'ELM_LABELS': None,
            'TP_TYPE': None,
            'R_T_S_ONLY': 'NO'
        }

        # Skipping the section of close-approach parameters.
        # Skipping the section of heliocentric ecliptic osculating elements.
        return requests.get(self.url, params=params)

    def get_ephemerides(self,
                        target: NonsiderealTarget,
                        overwrite: bool = False) -> EphemerisCoordinates:
        # TODO: ODB extractor must be mofidief.
        match target.tag:
            case TargetTag.COMET: horizons_name = f'NAME={target.des};CAP'
            case TargetTag.ASTEROID: horizons_name = f'ASTNAM={target.des};'
            case TargetTag.MAJORBODY: horizons_name = self.generate_horizons_id(target.des)
            # case _: raise ValueError(f'Unknown tag {target.tag}')
            case _: horizons_name = f'DES={target.des};'

        targ_name = target.des.replace(' ', '_').replace('/','')
        # end is the UT date, the same for both Gemini sites
        night_str = self.end.strftime(self.date_format)
        ephemeris_path = self.path / f'{self.site.name}_{targ_name}_{night_str}UT.eph'

        if not overwrite and ephemeris_path.exists() and ephemeris_path.is_file():
            logger.info(f'Reading ephemerides file for {target.des}')
            with ephemeris_path.open('r') as f:
                lines = [x.strip() for x in f.readlines()]
        else:
            logger.info(f'Querying JPL/Horizons for {horizons_name}')
            res = self._query(horizons_name)
            lines = res.text.splitlines()
            with ephemeris_path.open('w') as f:
                f.write(res.text)

        time = []
        coords = []

        try:
            firstline = lines.index('$$SOE') + 1
            lastline = lines.index('$$EOE') - 1

            for line in lines[firstline:lastline + 1]:
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
            logger.error(f'Error parsing ephemerides file for {target.des} at: {ephemeris_path}')
            raise e

        return EphemerisCoordinates(coordinates=coords, time=np.array(time))


@contextlib.contextmanager
def horizons_session(site: Site, start: datetime, end: datetime) -> ContextManager[HorizonsClient]:
    client = HorizonsClient(site=site, start=start, end=end)
    try:
        yield client
    finally:
        del client
    return
