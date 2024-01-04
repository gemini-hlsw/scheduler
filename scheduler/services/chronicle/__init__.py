# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import re
import os
from typing import Dict, Set, FrozenSet
from datetime import date, datetime, timedelta
from dataclasses import dataclass

from astropy.time import Time
from lucupy.sky import night_events
from lucupy.minimodel import ALL_SITES, Site

from definitions import ROOT_DIR
from scheduler.services.abstract import ExternalService


@dataclass(frozen=True)
class Interruption:
    """
    Parent class for any interruption in the night that would
    cause missing time of observation.
    """
    start: datetime
    time_loss: timedelta
    reason: str


@dataclass(frozen=True)
class Fault(Interruption):
    id: str


@dataclass(frozen=True)
class EngTask(Interruption):
    end: datetime


class ChronicleService(ExternalService):

    def __init__(self, sites: FrozenSet[Site] = ALL_SITES):
        self._sites = sites
        self._path = os.path.join(ROOT_DIR, 'scheduler', 'services', 'chronicle', 'data')
        # Fault reports by datetime to calculate missing instruments
        self._faults: Dict[Site, Dict[date, Set[Fault]]] = {site: {} for site in self._sites}

        # Engineering Task by datetime.
        self._eng_task: Dict[Site, Dict[date, Set[EngTask]]] = {site: {} for site in self._sites}


class FileBasedChronicle(ChronicleService):

    def _parse_eng_task_file(self, site: Site, to_file: str) -> None:
        """
        Parse Engineering task that block moments or the entire night.
        Each twilight is calculated using lucupy.sky, some discrepancies might affect.
        """
        pattern = r'(\[.*\])'
        # Ignore GS until the file is created.
        if site is Site.GS:
            return

        with open(os.path.join(self._path, to_file), 'r') as file:
            for line_num, line in enumerate(file):
                # Find pattern to keep bracket comments not split.
                match = re.search(pattern, line)
                if match:
                    comment = match.group(1)
                    rest_of_line = re.sub(pattern, '', line)
                    eng_date, start_time, end_time = rest_of_line.strip().split()
                    #  Single date for key
                    just_date = datetime.strptime(eng_date, '%Y-%m-%d')
                    # Time day in jd to calculate twilight
                    time = Time(just_date)
                    _, _, _, even_12twi, morn_12twi, _, _ = night_events(time,
                                                                         site.location,
                                                                         site.timezone)

                    # Handle twilight
                    if start_time == 'twi':
                        start_date = even_12twi.datetime
                        end_date = (morn_12twi.datetime if end_time == 'twi' else
                                    datetime.strptime(f'{eng_date} {end_time}', '%Y-%m-%d %H:%M'))
                    else:
                        start_date = datetime.strptime(f'{eng_date} {start_time}', '%Y-%m-%d %H:%M')
                        end_date = (morn_12twi.datetime if end_time == 'twi'
                                    else datetime.strptime(f'{eng_date} {end_time}', '%Y-%m-%d %H:%M'))

                    time_loss = end_date - start_date
                    eng_task = EngTask(start=start_date,
                                       end=end_date,
                                       time_loss=time_loss,
                                       reason=comment)

                    if just_date in self._eng_task[site]:
                        self._eng_task[site][just_date].add(eng_task)
                    else:
                        self._eng_task[site][just_date] = {eng_task}
                else:
                    raise ValueError(f'Pattern not found. Format error on Eng Task file at line {line_num}')

    def _parse_faults_file(self, site: Site, to_file: str) -> None:
        """Parse faults from files.
        This is purposeful left non-private as might be used with incoming files from
        the React app.
        """
        # Files contains this repetitive string in each timestamp, if we need them
        # we could add them as constants.
        ts_clean = ' 04:00' if site is Site.GS else ' 10:00'
        with open(os.path.join(self._path, to_file), 'r') as file:
            for line_num, original_line in enumerate(file):
                line = original_line.rstrip()  # remove trail spaces
                if line:  # ignore empty lines
                    if line[0].isdigit():
                        pass
                    elif line.startswith('FR'):  # found a fault
                        items = line.split('\t')
                        # Create timestamp with ts_clean var removed
                        ts = datetime.strptime(items[1].replace(ts_clean, ''),
                                               '%Y %m %d  %H:%M:%S')
                        fault = Fault(id=items[0],
                                      start=ts,  # date with time
                                      time_loss=timedelta(hours=float(items[2])),  # time loss
                                      reason=items[3])  # comment for the fault
                        if ts.date() in self._faults[site]:
                            self._faults[site][ts.date()].add(fault)
                        else:
                            self._faults[site][ts.date()] = {fault}
                    else:
                        raise ValueError(f'Fault file has wrong format at line {line_num}')
