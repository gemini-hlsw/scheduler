# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import contextlib
import dateutil.parser
import requests
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time as time_module
from pathlib import Path
from typing import final, ContextManager, Dict, Final

import numpy as np
from lucupy.helpers import dms2rad, hms2rad
from lucupy.minimodel import NonsiderealTarget, Semester, TargetTag, Site

from numpy import arctan2, cos, sin, sqrt
from scipy.interpolate import CubicSpline
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
    time_slot_length: float

    # We look up across the whole night, so the labels are simply night labels.
    date_format: str = field(default='%Y%m%d')
    path: Path = field(default=Path(ROOT_DIR) / 'scheduler' / 'services' / 'horizons' / 'data')
    url: str = field(default='https://ssd.jpl.nasa.gov/api/horizons.api')

    @staticmethod
    def generate_horizons_id(designation: str) -> str:
        des = designation.lower()
        return _MAJORBODY_DICT.get(des, des)

    def _query(self,
               target: str,
               start: str,
               stop: str,
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
            'format': 'text',
            'COMMAND': f"'{target}'",
            'OBJ_DATA': object_data,
            'MAKE_EPHEM': make_ephem,
            'EPHEM_TYPE': 'OBSERVER',
            'CENTER': center,
            'START_TIME': start,
            'STOP_TIME': stop,
            'STEP_SIZE': f"'{step}'",
            'QUANTITIES': quantities,
            'REF_SYSTEM': 'J2000',
            'CAL_FORMAT': cal_format,
            'ANG_FORMAT': 'HMS',
            'TIME_DIGITS': 'MINUTES',
            'SUPPRESS_RANGE_RATE': 'NO',
            'ELEV_CUT': '-90',
            'SKIP_DAYLT': skip_day,
            'SOLAR_ELONG': "'0,180'",
            'AIRMASS': 100,
            'EXTRA_PREC': 'YES',
            'CSV_FORMAT': csv_format,
            'R_T_S_ONLY': 'NO'
        }

        # Skipping the section of close-approach parameters.
        # Skipping the section of heliocentric ecliptic osculating elements.
        return requests.get(self.url, params=params)

    def get_coords_table(self, lines: list[str]):
        time = []
        coords = []
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

                time.append(dateutil.parser.parse(line[0:17]))
                coords.append(Coordinates(hms2rad(rah, ram, ras), dms2rad(decd, decm, decs, decg)))

        return time, coords

    def interpolate_coords(self, time_list: list[datetime], coord_list: list[Coordinates], timeslot_length: timedelta = timedelta(minutes=1.0)):
        """
        Interpolate coord_list at every timeslot between the first and last entry of time_list.

        Fits a cubic spline through the unit vectors of the whole coord_list sequence (rather than
        linearly/spherically interpolating each consecutive pair in isolation), so the result
        follows the curvature of the target's actual path instead of a piecewise-linear one.
        """
        times64 = np.array(time_list, dtype='datetime64[us]')
        step_us = int(timeslot_length / timedelta(microseconds=1))
        gaps_us = (np.diff(times64) / np.timedelta64(1, 'us')).astype(np.int64)
        n_per_segment = gaps_us // step_us

        total = int(n_per_segment.sum())
        starts = np.concatenate(([0], np.cumsum(n_per_segment)[:-1]))
        step_index = np.arange(total) - np.repeat(starts, n_per_segment)

        seg_start_times = np.repeat(times64[:-1], n_per_segment)
        interpolated_times64 = seg_start_times + (step_index * step_us).astype('timedelta64[us]')

        knot_us = (times64.astype(np.int64) - times64[0].astype(np.int64)).astype(np.float64)
        query_us = (interpolated_times64.astype(np.int64) - times64[0].astype(np.int64)).astype(np.float64)

        ra = np.array([c.ra for c in coord_list])
        dec = np.array([c.dec for c in coord_list])
        unit_vectors = np.stack([cos(dec) * cos(ra), cos(dec) * sin(ra), sin(dec)], axis=1)

        spline = CubicSpline(knot_us, unit_vectors, axis=0)
        interpolated_vectors = spline(query_us)
        interpolated_vectors /= np.linalg.norm(interpolated_vectors, axis=1, keepdims=True)

        x, y, z = interpolated_vectors[:, 0], interpolated_vectors[:, 1], interpolated_vectors[:, 2]
        new_ra = arctan2(y, x)
        new_dec = arctan2(z, sqrt(x ** 2 + y ** 2))

        interpolated_coords = [Coordinates(float(r), float(d)) for r, d in zip(new_ra, new_dec)]
        interpolated_times = interpolated_times64.astype('datetime64[us]').astype(datetime).tolist()

        return interpolated_times, interpolated_coords

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
        semester = Semester.find_semester_from_date(self.start)
        ephemeris_path = self.path / f'{self.site.name}_{targ_name}_{str(semester)}.eph'

        lines = None
        if not overwrite and ephemeris_path.exists() and ephemeris_path.is_file():
            logger.debug(f'Reading ephemerides file for {target.des}')
            with ephemeris_path.open('r') as f:
                cached_lines = [x.strip() for x in f.readlines()]
            # Cached file is only usable if it contains the SOE/EOE markers.
            # Old failed queries (e.g. from a wrong COMMAND param) leave error
            # pages on disk; treat those as a cache miss and re-fetch instead
            # of letting the parser raise.
            if '$$SOE' in cached_lines and '$$EOE' in cached_lines:
                lines = cached_lines
            else:
                logger.warning(
                    f'Cached ephemerides for {target.des} at {ephemeris_path} '
                    'is missing $$SOE/$$EOE markers; re-fetching from Horizons.'
                )

        if lines is None:
            semester_start = semester.start_date().strftime("'%Y-%b-%d %H:%M'")
            semester_end = semester.end_date().strftime("'%Y-%b-%d %H:%M'")
            logger.debug(f'Querying JPL/Horizons for {horizons_name}')
            res = self._query(horizons_name,
                              semester_start,
                              semester_end,
                              daytime=True,
                              step='4h')
            lines = res.text.splitlines()
            with ephemeris_path.open('w') as f:
                f.write(res.text)

        try:
            time, coords = self.get_coords_table(lines)
            start_index = max(0, next(i for i, t in enumerate(time) if t >= self.start) - 1)
            end_index = next(i for i, t in enumerate(time) if t >= self.end)
            time, coords = time[start_index:end_index], coords[start_index:end_index]
            time, coords = self.interpolate_coords(time, coords, timedelta(minutes=self.time_slot_length))

        except ValueError as e:
            logger.error(f'Error parsing ephemerides file for {target.des} at: {ephemeris_path}')
            raise e

        return EphemerisCoordinates(coordinates=coords, time=np.array(time))


@contextlib.contextmanager
def horizons_session(site: Site, start: datetime, end: datetime, time_slot_length: float) -> ContextManager[HorizonsClient]:
    client = HorizonsClient(site=site, start=start, end=end, time_slot_length=time_slot_length)
    try:
        yield client
    finally:
        del client
    return
