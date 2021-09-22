import logging
from zipfile import ZipFile
import xml.etree.cElementTree as ElementTree

import astropy.coordinates
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table, vstack

import pytz
import os
import numpy as np

import collector.sb as sb
from collector.vskyutil import nightevents
from collector.xmlutils import *
from collector.get_tadata import get_report, get_tas, sumtas_date
from collector.program import Program

from common.constants import FUZZY_BOUNDARY, CLASSICAL_NIGHT_LEN
from common.helpers import round_min
from common.structures.conditions import SkyConditions, WindConditions, conditions_parser, IQ, CC, SB, WV
from common.structures.elevation import ElevationConstraints, str_to_elevation_type, str_to_float
from common.structures.site import Site, GEOGRAPHICAL_LOCATIONS, SITE_ZIPS
from common.structures.target import TargetTag, Target
from common.structures.time_award_units import TimeAwardUnits

from common.structures.band import Band
from common.structures.instrument import Instrument
from greedy_max.schedule import Observation
from greedy_max.category import Category

from typing import List, Dict, Optional, NoReturn, Iterable, FrozenSet

INFINITE_DURATION = 3. * 365. * 24. * u.h  # A date or duration to use for infinity (length of LP)
INFINITE_REPEATS = 1000  # number to depict infinity for repeats in OT Timing windows calculations
MIN_NIGHT_EVENT_TIME = Time('1980-01-01 00:00:00', format='iso', scale='utc')
MAX_NIGHT_EVENT_TIME = Time('2200-01-01 00:00:00', format='iso', scale='utc')


def ot_timing_windows(starts: Iterable[int],
                      durations: Iterable[int],
                      repeats: Iterable[int],
                      periods: Iterable[int]) -> List[Time]:
    """
    Turn OT timing constraints into more natural units
    Inputs are lists
    Match output from GetWindows
    """

    timing_windows = []

    for (start, duration, repeat, period) in zip(starts, durations, repeats, periods):

        # The timestamps are in milliseconds
        # The start time is unix time (milliseconds from 1970-01-01 00:00:00) UTC
        # Time requires unix time in seconds
        t0 = float(start) * u.ms
        begin = Time(t0.to_value('s'), format='unix', scale='utc')

        # duration = -1 means forever
        duration = INFINITE_DURATION if duration == -1 else duration / 3600000. * u.h

        # repeat = -1 means infinite, and we require at least one repeat
        repeat = INFINITE_REPEATS if repeat == -1 else max(1, repeat)

        # period between repeats
        period = period / 3600000. * u.h

        for i in range(repeat):
            window_start = begin + i * period
            window_end = window_start + duration
            timing_windows.append(Time([window_start, window_end]))

    return timing_windows


class Collector:
    # Counter to keep track of observations.
    observation_num = 0

    def __init__(self,
                 sites: FrozenSet[Site],
                 semesters: FrozenSet[str],
                 program_types: FrozenSet[str],
                 obs_classes: FrozenSet[str],
                 time_range: Time = None,
                 delta_time: Time = 1.0 * u.min):
        
        self.sites = sites                   
        self.semesters = semesters
        self.program_types = program_types
        self.obs_classes = obs_classes

        self.time_range = time_range  # Time object: array for visibility start/stop dates.
        self.time_grid = self._calculate_time_grid()  # Time object: array with entry for each day in time_range.
        self.delta_time = delta_time  # Length of time steps.

        self.observations = []
        self.programs = {}

        self.night_events = {}        
        self.scheduling_groups = {}
        self.inst_config = []
        
        # NOTE: This are used to used the EarthLocation and Timezone objects for functions that used those kind of 
        # objects. This can either be include in the Site class if the use of this libraries is justified. 
        self.timezones = {}
        self.locations = {}

        self.programs = {}
        self.obsid = []
        self.obstatus = []
        self.obsclass = []
        self.observation_num = 0
        self.band = []
        self.toostatus = []
        self.priority = []

        self.tot_time = []  # total program time
        self.obs_time = []  # used/scheduled time
        self.obs_windows = []

    def load(self, path: str) -> NoReturn:
        """ Main collector method. It setups the collecting process and parameters """ 

        # TODO: temporary hack for just using one site
        site_name = Site.GS.value

        xmlselect = [site_name.upper() + '-' + sem + '-' + prog_type
                     for sem in self.semesters for prog_type in self.program_types]

        # Retrieve the site details
        try:
            site = Site(site_name)
            location = GEOGRAPHICAL_LOCATIONS[site]
        except ValueError:
            raise RuntimeError(f'{site_name} not a valid site name.')
        except astropy.coordinates.UnknownSiteException:
            raise RuntimeError("${site.value} not a valid geographical location.")

        self.timezones[site] = pytz.timezone(location.info.meta['timezone'])
        self.locations[site] = location

        self._calculate_night_events()

        # TODO: We will have to modify in order for this code to be usable by other observatories.
        zip_path = os.path.join(path, f'{(self.time_range[0] - 1.0 * u.day).strftime("%Y%m%d")}{SITE_ZIPS[site]}')
        logging.info(f'Retrieving program data from: {zip_path}.')

        time_accounting = self._load_tas(path, site)
        self._readzip(zip_path, xmlselect, site_name, tas=time_accounting, obsclasses=self.obs_classes)

    def _calculate_time_grid(self) -> Optional[Time]:
        if self.time_range is not None:
            # Add one day to make the time_range inclusive since using arange.
            return Time(np.arange(self.time_range[0].jd, self.time_range[1].jd + 1.0, (1.0 * u.day).value), format='jd')
        return None

    def _calculate_night_events(self) -> NoReturn:
        """Load night events to collector"""

        if self.time_grid is not None:

            for site in self.sites:
                tz = self.timezones[site]
                site_location = self.locations[site]
                mid, sset, srise, twi_eve18, twi_mor18, twi_eve12, twi_mor12, mrise, mset, smangs, moonillum = \
                    nightevents(self.time_grid, site_location, tz, verbose=False)
                night_length = (twi_mor12 - twi_eve12).to_value('h') * u.h
                self.night_events[site] = {'midnight': mid, 'sunset': sset, 'sunrise': srise,
                                                            'twi_eve18': twi_eve18, 'twi_mor18': twi_mor18,
                                                            'twi_eve12': twi_eve12, 'twi_mor12': twi_mor12,
                                                            'night_length': night_length,
                                                            'moonrise': mrise, 'moonset': mset,
                                                            'sunmoonang': smangs, 'moonillum': moonillum}

    def _load_tas(self, path: str, site: Site) -> Dict[str, Dict[str, float]]:
        """Load Time Accounting Summary."""

        date = self.time_range[0].strftime('%Y%m%d')
        plan_path = os.path.join(path, 'nightplans', date)
        if not os.path.exists(plan_path):
            os.makedirs(plan_path)
        logging.info(f"Using night plan directory: {plan_path}.")

        tas = Table()
        tadate = (self.time_range[0] - 1.0 * u.day).strftime('%Y%m%d')
        
        for sem in self.semesters:
            ta_file = f'tas_{site.name}_{sem}.txt'
            ta_file_path = os.path.join(plan_path, ta_file)
            if not os.path.exists(ta_file_path):
                get_report(site, ta_file, plan_path)
                
            tmp = get_tas(ta_file_path)
            tas = vstack([tas, tmp])

        return sumtas_date(tas, tadate)

    @staticmethod
    def _instrument_setup(configuration: Dict[str, List[str]], instrument_name: str) -> Instrument:
        """ Setup instrument configurations for each observation """
        instconfig = {} 

        fpuwidths = []
        disperser = 'NONE'
        for key in configuration.keys():
            ulist = list(dict.fromkeys(configuration[key]))
            if key in ['filter', 'disperser']:
                for kk in range(len(ulist)):
                    ifnd = ulist[kk].find('_')
                    if ifnd == -1:
                        ifnd = len(ulist[kk])
                ulist[kk] = ulist[kk][0:ifnd]

            # Use human-readable slit names
            if 'GMOS' in instrument_name:
                if key == 'fpu':
                    fpulist = FpuXmlTranslator(ulist)
                    fpunames = []
                    for fpu in fpulist:
                        fpunames.append(fpu['name'])
                        fpuwidths.append(fpu['width'])
                    ulist = fpunames
                    instconfig['fpuWidth'] = fpuwidths
                if key == 'customSlitWidth':
                    for cwidth in ulist:
                        fpuwidths.append(CustomMaskWidth(cwidth))
                    instconfig['fpuWidth'] = fpuwidths
            if key == 'disperser':
                disperser = ulist[0]
            else:
                instconfig[key] = ulist

        # TODO: I expect this can be simplified.
        if any(inst in instrument_name.upper() for inst in ['IGRINS', 'MAROON-X']):
            disperser = 'XD'
       
        return Instrument(instrument_name, disperser, instconfig)
        
    def _readzip(self, 
                 zipfile: str, 
                 xmlselect: List[str], 
                 site: Site,
                 selection=frozenset(['ONGOING', 'READY']),
                 obsclasses=frozenset(['SCIENCE']),
                 tas=None):
        """ Populate Database from the zip file of an ODB backup """

        with ZipFile(zipfile, 'r') as zip:
            names = zip.namelist()
            names.sort()
            
            for name in names:
                if any(xs in name for xs in xmlselect):
                    tree = ElementTree.fromstring(zip.read(name))
                    program = tree.find('container')
                    (active, complete) = CheckStatus(program)
                    if active and not complete:
                        self._process_observation_data(program, selection, obsclasses, tas)
   
    def _process_observation_data(self,
                                  program_data,
                                  selection: frozenset[str],
                                  obsclasses: frozenset[str],
                                  tas: Dict[str, Dict[str, float]]) -> NoReturn:
        """Parse XML file to Observation objects and other data structures"""

        program_id = get_program_id(program_data)
        notes = get_program_notes(program_data)
        program_mode = get_program_mode(program_data)
        
        xml_band = get_program_band(program_data)
        if xml_band == 'UNKNOWN':
            band = Band(1) if program_mode == 'CLASSICAL' else Band(0)
        else:
            band = Band(int(xml_band))

        award, unit = get_program_awarded_time(program_data)
        if award and unit:
            award = CLASSICAL_NIGHT_LEN * float(award) * u.hour if unit == TimeAwardUnits.NIGHTS else float(award) * u.hour
        else:
            award = 0.0 * u.hour
        
        year = program_id[3:7]
        next_year = str(int(year) + 1)
        semester = program_id[7]

        if 'FT' in program_id:
            program_start, program_end = GetFTProgramDates(notes, semester, year, next_year)
            # If still undefined, use the values from the previous observation
            if program_start is None:
                program_start = self.programs[-1].start
                program_end = self.programs[-1].end

        else:
            beginning_semester_1 = Time(year + "-02-01 20:00:00", format='iso')
            end_semester_1 = Time(year + "-08-01 20:00:00", format='iso')
            beginning_semester_2 = Time(next_year + "-02-01 20:00:00", format='iso')
            end_semester_2 = Time(next_year + "-08-01 20:00:00", format='iso')

            # This covers 'Q', 'LP' and 'DD' program observations
            # Note that Band 1 non-ToO programs are persistent for the following semester
            if semester == 'A':
                program_start = beginning_semester_1
                program_end = beginning_semester_2 if band == Band.Band1 else end_semester_1
            else:
                program_start = end_semester_1
                program_end = end_semester_2 if band == Band.Band1 else beginning_semester_2

        # Flexible boundaries - could be type-dependent
        program_start -= FUZZY_BOUNDARY * u.day
        program_end += FUZZY_BOUNDARY * u.day

        # Thesis program?
        thesis = GetThesis(program_data)

        # ToO status
        toostat = get_too_status(program_data)

        # Used time from Time Accounting Summary (tas) information
        if tas and program_id in tas:
            used = tas[program_id]['prgtime']
        else:
            used = 0.0 * u.hour

        collected_program = Program(program_id,
                                    program_mode,
                                    band,
                                    thesis,
                                    award,
                                    used,
                                    toostat,
                                    program_start,
                                    program_end)
        self.programs[program_id] = collected_program
        raw_observations, groups = GetObservationInfo(program_data)

        if raw_observations is None:
            raise RuntimeError('Parser issue to get observation info')

        for raw_observation, group in zip(raw_observations, groups):

            classes = list(dict.fromkeys(GetClass(raw_observation)))
            status = GetObsStatus(raw_observation)
            obs_odb_id = GetObsID(raw_observation)

            if any(obs_class in obsclasses for obs_class in classes) and (status in selection):
                
                print('Adding ' + obs_odb_id, end='\r')
                total_time = GetObsTime(raw_observation)
                
                #Target Info
                target_name, ra, dec = GetTargetCoords(raw_observation) 
                
                if target_name is None:
                    target_name = 'None'
                if ra is None and dec is None:
                    ra = 0.0
                    dec = 0.0
                target_coords = SkyCoord(ra, dec, frame='icrs', unit=(u.deg, u.deg))
                target_mags = GetTargetMags(raw_observation, baseonly=True)
                targets = GetTargets(raw_observation)
                target_designation = None
                target_tag = None
                if targets:
                    for target in targets:
                        try:
                            target_name = target['group']['name']
                            target_tag = TargetTag(target['tag']) if target_name == 'Base' else None
                            target_designation = target['num'] if target_tag is not None and\
                                                                  target_tag == TargetTag.MajorBody else target['des']
                        except:
                            pass

                target = Target(target_name, target_tag, target_mags, target_designation, target_coords)
                # Observation Priority
                priority = GetPriority(raw_observation)
                # Instrument Configuration
                instrument_name = GetInstrument(raw_observation)
                instrument_config = GetInstConfigs(raw_observation)
                if 'name' in instrument_config:
                    instrument_name = instrument_config['name'][0]
                
                # ToO status
                too_status = GetObsTooStatus(raw_observation, self.programs[program_id].too_status)
                
                # Sky Conditions
                conditions = GetConditions(raw_observation, label=False)
                
                if conditions or conditions is None:
                    sky_cond = SkyConditions() 
                else:               
                    parse_conditions = conditions_parser(conditions)
                    sky_cond = SkyConditions(*parse_conditions)
                
                # Elevation constraints        
                elevation_type, min_elevation, max_elevation = GetElevation(raw_observation)
                elevation_type = str_to_elevation_type(elevation_type)
                min_elevation, max_elevation = str_to_float(min_elevation), str_to_float(max_elevation)
                elevation_constraints = ElevationConstraints(elevation_type, min_elevation, max_elevation)
                
                # Charged observation time
                # Used time from Time Accounting Summary (tas) information
                obs_time = 0
                if tas and program_id in tas and obs_odb_id in tas[program_id]:
                    obs_time = max(0, tas[program_id][obs_odb_id]['prgtime'].value)\

                # Total observation time
                # for IGRINS, update tot_time to take telluric into account if there is time remaining
                calibration_time = 0
                if 'IGRINS' in instrument_name.upper() and total_time.total_seconds() / 3600. - obs_time > 0.0:
                    calibration_time = 10/ 60   # fractional hour

                inst_config = Collector._instrument_setup(instrument_config,instrument_name)
                
                start, duration, repeat, period = GetWindows(raw_observation) 
                # This makes a list of timing windows between progstart and progend
                timing_windows = ot_timing_windows(start, duration, repeat, period)

                progstart = collected_program.start
                progend = collected_program.end

                windows = []
                if len(timing_windows) == 0:
                    windows.append(Time([progstart, progend]))
                else:
                    for ii in range(len(timing_windows)):
                        if (timing_windows[ii][0] <= progend) and \
                                (timing_windows[ii][1] >= progstart):
                            wstart = max(progstart, timing_windows[ii][0])
                            wend = min(progend, timing_windows[ii][1])
                        else:
                            wstart = progstart
                            wend = progstart
                        # Timing window starts at the beginning of the sequence, the slew can be outside the window
                        # Therefore, subtract acquisition time from start of timing window
                        wstart -= self.observations[-1].acquisition()
                        windows.append(Time([round_min(wstart), round_min(wend)]))
                self.obs_windows.append(windows)
                
                # Observation number in program
                collected_program.add_observation(self.observation_num)


                # Check if group exists, add if not
                if group['key'] not in self.scheduling_groups.keys():
                    self.scheduling_groups[group['key']] = {'name': group['name'], 'idx': []}
                    collected_program.add_group(group['key'])
                self.scheduling_groups[group['key']]['idx'].append(Collector.observation_num)

                # Get Horizons coordinates for nonsidereal targets (write file for future use)
                #if target_tag in ['asteroid', 'comet', 'major-body']: NOTE: This does not work, and it should!
                
                self.obstatus.append(status)
                #self.target_name.append(target_name)
                #self.target_tag.append(target_tag)
                #self.target_des.append(des)
                #self.coord.append(SkyCoord(ra, dec, frame='icrs', unit=(u.deg, u.deg)))
                #self.mags.append(target_mags)
                self.toostatus.append(toostat.lower())
                self.priority.append(priority)
                #self.conditions.append({'iq': condf[0], 'cc': condf[1], 'bg': condf[2], 'wv': condf[3]})

                self.observations.append(Observation(Collector.observation_num,
                                                     obs_odb_id,
                                                     band,
                                                     Category(classes[0].lower()),
                                                     obs_time,
                                                     total_time.total_seconds() / 3600. + calibration_time,
                                                     inst_config,
                                                     sky_cond,
                                                     elevation_constraints,
                                                     target,
                                                     status,
                                                     too_status.lower()))
                Collector.observation_num += 1

    def create_time_array(self):

        timesarr = []

        for i in range(len(self.time_grid)):
            
            tmin = min([MAX_NIGHT_EVENT_TIME] + [self.night_events[site]['twi_eve12'][i] for site in self.sites])
            tmax = max([MIN_NIGHT_EVENT_TIME] + [self.night_events[site]['twi_mor12'][i] for site in self.sites])

            tstart = round_min(tmin, up=True)
            tend = round_min(tmax, up=False)
            n = np.int((tend.jd - tstart.jd) / self.delta_time.to(u.day).value + 0.5)
            times = Time(np.linspace(tstart.jd, tend.jd - self.delta_time.to(u.day).value, n), format='jd')
            timesarr.append(times)

        return timesarr
    
    def get_actual_conditions(self):
        # TODO: This could be an static method but it should be some internal process or API call to ENV
        # but right now is mostly hardcoded so the plan would be as if 

        actcond = {}
        time_blocks = [Time(["2021-04-24 04:30:00", "2021-04-24 08:00:00"], format='iso', scale='utc')] #
        variants = {
            #             'IQ20 CC50': {'iq': 0.2, 'cc': 0.5, 'wv': 1.0, 'wd': -1, 'ws': -1},
            'IQ70 CC50': {'iq': IQ.IQ70, 'cc': CC.CC50, 'wv': WV.WVANY, 'wdir': 330.*u.deg, 'wsep': 40.*u.deg, 'wspd': 5.0*u.m/u.s, 'tb': time_blocks},
            #             'IQ70 CC70': {'iq': 0.7, 'cc': 0.7, 'wv': 1.0, 'wdir': -1, 'wsep': 30.*u.deg, 'wspd': 0.0*u.m/u.s, 'tb': time_blocks}, 
            #             'IQ85 CC50': {'iq': 0.85, 'cc': 0.5, 'wv': 1.0, 'wdir': -1, 'wsep': 30.0*u.deg, 'wspd': 0.0*u.m/u.s, 'tb': time_blocks}, 
            #             'IQ85 CC70': {'iq': 0.85, 'cc': 0.7, 'wv': 1.0, 'wdir': -1, 'wsep': 30.0*u.deg, 'wspd': 0.0*u.m/u.s, 'tb': time_blocks},
            #             'IQ85 CC80': {'iq': 1.0, 'cc': 0.8, 'wv': 1.0, 'wdir': -1, 'wsep': 30.0*u.deg, 'wspd': 0.0*u.m/u.s, 'tb': []}
            }

        selected_variant = variants['IQ70 CC50']
        
        actcond['sky'] = SkyConditions(selected_variant['iq'], SB.SBANY, selected_variant['cc'],
                             selected_variant['wv'])
        actcond['wind'] = WindConditions( selected_variant['wsep'], selected_variant['wspd'],
                             selected_variant['wdir'],time_blocks)
        
        return actcond