import re
import os
from io import BytesIO
from typing import Dict, FrozenSet, Optional, List
from datetime import date, datetime, timedelta, time
from dataclasses import dataclass
from typing import final, Union

from lucupy.minimodel import ALL_SITES, Site

from definitions import ROOT_DIR
from scheduler.services.abstract import ExternalService


@final
@dataclass
class TwilightTime:
    value: str = "twi"

    def __sub__(self, other):
        # When an unresolved twilight is calculated
        # show a huge number to indentify error.
        return 9999


@dataclass
class Interruption:
    """
    Parent class for any interruption in the night that would
    cause missing time of observation.
    """
    start: Union[datetime, TwilightTime]
    reason: str
    id: Optional[str] = None  # For FR only
    end: Optional[Union[datetime, TwilightTime]] = None
    time_loss: Optional[timedelta] = None

    def resolve_twilight(self, even_twi, morn_twi):
        """
        Resolve time_loss when values are TwilightTime.
        This has to be called in the Collector.
        """
        if self.start is TwilightTime:
            self.start = even_twi.datetime
        if self.end is TwilightTime:
            self.end = morn_twi.datetime
        self.time_loss = self.end - self.start


@dataclass
class Fault(Interruption):
    pass


@dataclass
class EngTask(Interruption):
    pass


@dataclass
class WeatherLoss(Interruption):
    pass


class ChronicleService(ExternalService):

    def __init__(self, sites: FrozenSet[Site] = ALL_SITES):
        self._sites = sites
        self._path = os.path.join(ROOT_DIR, 'scheduler', 'services', 'chronicle', 'data')
        # Fault reports by datetime to calculate missing instruments
        self._faults: Dict[Site, Dict[date, List[Fault]]] = {site: {} for site in self._sites}

        # Engineering Task by datetime.
        self._eng_task: Dict[Site, Dict[date, List[EngTask]]] = {site: {} for site in self._sites}
        self._weather_loss: Dict[Site, Dict[date, List[WeatherLoss]]] = {site: {} for site in self._sites}


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
                    # Handle twilight
                    if start_time == 'twi':
                        start_date = TwilightTime()
                        end_date = (TwilightTime() if end_time == 'twi' else
                                    datetime.strptime(f'{eng_date} {end_time}', '%Y-%m-%d %H:%M'))
                    else:
                        start_date = datetime.strptime(f'{eng_date} {start_time}', '%Y-%m-%d %H:%M')
                        end_date = (TwilightTime() if end_time == 'twi'
                                    else datetime.strptime(f'{eng_date} {end_time}', '%Y-%m-%d %H:%M'))

                    eng_task = EngTask(start=start_date,
                                       end=end_date,
                                       reason=comment)

                    if just_date in self._eng_task[site]:
                        self._eng_task[site][just_date].append(eng_task)
                    else:
                        self._eng_task[site][just_date] = [eng_task]
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
                        # TODO: Unused variable and just a string, not a semester.
                        semester = line
                    elif line.startswith('FR'):  # found a fault
                        items = line.split('\t')
                        # Create timestamp with ts_clean var removed
                        ts = datetime.strptime(items[1].replace(ts_clean, ''),
                                               '%Y %m %d  %H:%M:%S')
                        fault = Fault(id=items[0],
                                      start=ts,  # date with time
                                      reason=items[3],  # comment for the fault
                                      time_loss=timedelta(hours=float(items[2])))  # time loss
                        if ts.date() in self._faults[site]:
                            self._faults[site][ts.date()].append(fault)
                        else:
                            self._faults[site][ts.date()] = [fault]
                    else:
                        raise ValueError(f'Fault file has wrong format at line {line_num}')

    def _parse_weather_time_lost(self, site: Site, to_file: str) -> None:
        with open(os.path.join(self._path, to_file), 'r') as file:
            try:
                for line_num, line in enumerate(file):
                    data = line.split()

                    start_date = datetime.strptime(data[0], "%Y-%m-%d")
                    end_date = datetime.strptime(data[0], "%Y-%m-%d")
                    start_time = data[1]
                    end_time = data[2]

                    # Handle twilight
                    if start_time == 'twi':
                        start = TwilightTime()
                        if end_time == 'twi':
                            end = TwilightTime()
                        else:
                            end_ts = datetime.strptime(data[2], "%H:%M")
                            if end_ts.time() > time(0, 0):
                                end_date += timedelta(days=1)
                            end = datetime.combine(end_date, end_ts.time())
                    else:
                        start_ts = datetime.strptime(data[1], "%H:%M")
                        if start_ts.time() > time(0, 0):
                            start_date += timedelta(days=1)
                        start = datetime.combine(start_date, start_ts.time())
                        if end_time == 'twi':
                            end = TwilightTime()
                        else:
                            end_ts = datetime.strptime(data[2], "%H:%M")
                            if end_ts.time() > time(0, 0):
                                end_date += timedelta(days=1)
                            end = datetime.combine(end_date, end_ts.time())

                    msg = data[3] if len(data) == 4 else ""
                    weather_loss = WeatherLoss(start=start,
                                               end=end,
                                               reason=msg)

                    if start_date.date() in self._weather_loss[site]:
                        self._weather_loss[site][start_date.date()].append(weather_loss)
                    else:
                        self._weather_loss[site][start_date.date()] = [weather_loss]
            except ValueError:
                raise ValueError(f'Problem parsing line {line_num}')

    def load_files(self,
                   site: Site,
                   eng_task_file: Union[str, BytesIO],
                   faults_file: Union[str, BytesIO],
                   weather_loss_file: Union[str, BytesIO]):
        # Load Engineering task files
        self._parse_eng_task_file(site, eng_task_file)
        # Load Faults history file
        self._parse_faults_file(site, faults_file)
        # Load Loss time for Weather file
        self._parse_weather_time_lost(site,  weather_loss_file)


class OcsChronicleService(FileBasedChronicle):
    def __init__(self, sites: FrozenSet[Site] = ALL_SITES):
        super().__init__(sites)

        for site in self._sites:
            suffix = ('s' if site == Site.GS else 'n').upper()
            self.load_files(site,
                            f'EngTasksG{suffix}.txt',
                            f'Faults_AllG{suffix}.txt',
                            f'WLG{suffix}.txt')


class FileChronicleService(FileBasedChronicle):
    def __init__(self, sites: FrozenSet[Site] = ALL_SITES):
        super().__init__(sites)
