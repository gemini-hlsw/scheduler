# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from datetime import datetime, timedelta
import dateutil.parser
from pathlib import Path
from definitions import ROOT_DIR
from scheduler.services.horizons.coordinates import Coordinates
import matplotlib.pyplot as plt

from lucupy.helpers import dms2rad, hms2rad

import numpy as np
from numpy import arctan2, cos, sin, sqrt
from scipy.interpolate import CubicSpline

def interpolate_coords(time2: list[datetime], coords2: list[Coordinates], timeslot_length: timedelta = timedelta(minutes=1.0)):
    """
    Interpolate coords2 at every timeslot between the first and last entry of time2.

    Fits a cubic spline through the unit vectors of the whole coords2 sequence (rather than
    linearly/spherically interpolating each consecutive pair in isolation), so the result
    follows the curvature of the target's actual path instead of a piecewise-linear one.
    """
    times64 = np.array(time2, dtype='datetime64[us]')
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

    ra = np.array([c.ra for c in coords2])
    dec = np.array([c.dec for c in coords2])
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


def get_coords_table(lines: list[str]):
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

file_path = Path(ROOT_DIR) / 'scheduler' / 'services' / 'horizons' / 'data'
file1 = file_path / 'GN_2018_PM20_20181021UT.eph'
file2 = file_path / 'GN_2018_PM20_20181022UT.eph'
file3 = file_path / 'GN_2018_PM20_20181023UT.eph'
file4 = file_path / 'GN_2018_PM20_20181024UT.eph'
file5 = file_path / 'GN_2018_PM20_20181025UT.eph'
file6 = file_path / 'GN_2018_PM20_20181026UT.eph'
file7 = file_path / 'GN_2018_PM20_20181027UT.eph'
file8 = file_path / 'GN_2018_PM20_20181028UT.eph'
file_semester = file_path / 'GN_2018_PM20_2018B.eph'

date_files = [file1, file2, file3, file4, file5, file6, file7, file8]

time = []
coords = []
for file in date_files:
    with file.open('r') as f:
        lines = [x.strip() for x in f.readlines()]
    date_time, date_coords = get_coords_table(lines)
    time.extend(date_time)
    coords.extend(date_coords)

with file_semester.open('r') as f:
    semester_lines = [x.strip() for x in f.readlines()]

time2, coords2 = get_coords_table(semester_lines)
interpolated_times, interpolated_coords = interpolate_coords(time2, coords2, timedelta(minutes=1.0))

first_time = interpolated_times.index(time[0])
last_time = interpolated_times.index(time[-1])

first_sem_time = None
last_sem_time = None
for t in range(len(time2)):
    if time2[t] >= time[0]:
        first_sem_time = t
        break

for t in range(len(time2)):
    if time2[t] >= time[-1]:
        last_sem_time = t
        break


plt.figure(1)
plt.plot(time, [coord.ra for coord in coords], label='jpl 1m')
plt.plot(time2[first_sem_time:last_sem_time], [coord.ra for coord in coords2[first_sem_time:last_sem_time]], label='jpl 4h')
plt.plot(interpolated_times[first_time:last_time], [coord.ra for coord in interpolated_coords[first_time:last_time]], label='interpolated')
plt.legend()
plt.title('RA comparison')

plt.figure(2)
plt.plot(time, [coord.dec for coord in coords], label='jpl')
plt.plot(time2[first_sem_time:last_sem_time], [coord.dec for coord in coords2[first_sem_time:last_sem_time]], label='jpl 4h')
plt.plot(interpolated_times[first_time:last_time], [coord.dec for coord in interpolated_coords[first_time:last_time]], label='interpolated')
plt.legend()
plt.title('Dec comparison')

plt.show()