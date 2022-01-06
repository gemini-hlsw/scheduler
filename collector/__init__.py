from zipfile import ZipFile
import xml.etree.cElementTree as ElementTree

from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack

import os

import collector.sb as sb
from collector.vskyutil import nightevents
from collector.xmlutils import *
from collector.get_tadata import get_report, get_tas, sumtas_date
from collector.program import Program

from common.constants import FUZZY_BOUNDARY
from common.helpers.helpers import round_min
from common.structures.conditions import *
from common.structures.elevation import ElevationConstraints, str_to_elevation_type, str_to_float
from common.structures.site import Site, GEOGRAPHICAL_LOCATIONS, SITE_ZIP_EXTENSIONS, TIME_ZONES
from common.structures.target import TargetTag, Target

from common.structures.band import Band
from common.structures.instrument import GMOSConfiguration, Instrument, WavelengthConfiguration
from common.structures.obs_class import ObservationClass
from greedy_max.schedule import Observation

from typing import List, Dict, Optional, NoReturn, Iterable

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


def select_obsclass(classes: List[str])-> str:
    """Return the obsclass based on precedence

        classes: list of observe classes from get_obs_class
    """
    obsclass = ''

    # Precedence order for observation classes.
    obsclass_order = ['SCIENCE', 'PROG_CAL', 'PARTNER_CAL', 'ACQ', 'ACQ_CAL']

    # Set the obsclass for the entire observation based on obsclass precedence
    for oclass in obsclass_order:
        if oclass in classes:
            obsclass = oclass
            break

    return obsclass


class Collector:
    # Counter to keep track of observations.
    observation_num = 0

    def __init__(self,
                 sites: List[Site],
                 semesters: List[str],
                 program_types: List[str],
                 obs_classes: List[str],
                 time_range: Time = None,
                 time_slot_length: Time = 1.0 * u.min):

        self.sites = sites
        self.semesters = semesters
        self.program_types = program_types
        self.obs_classes = obs_classes

        self.time_range = time_range  # Time object: array for visibility start/stop dates.
        self.time_grid = self._calculate_time_grid()  # Time object: array with entry for each day in time_range.
        self.time_slot_length = time_slot_length  # Length of time steps.

        self.observations = {site: [] for site in self.sites}
        self.programs = {site: {} for site in self.sites}
        self.obs_windows = {site: [] for site in self.sites}
        self.scheduling_groups = {site: {} for site in self.sites}
        self.night_events = {}

    def load(self, path: str) -> NoReturn:
        """Main collector method. It sets up the collecting process and parameters."""
        for site in self.sites:
            site_name = site.value

            xmlselect = [site_name.upper() + '-' + sem + '-' + prog_type
                        for sem in self.semesters for prog_type in self.program_types]

            # TODO: We will have to modify in order for this code to be usable by other observatories.
            zip_path = os.path.join(path,
                                    f'{(self.time_range[0] - 1.0 * u.day).strftime("%Y%m%d")}{SITE_ZIP_EXTENSIONS[site]}')
            logging.info(f'Retrieving program data from: {zip_path}.')

            time_accounting = self._load_tas(path, site)
            self._calculate_night_events(site)
            self._readzip(zip_path, xmlselect, site, tas=time_accounting)
            Collector.observation_num = 0

            logging.info(f'Added {len(self.observations[site])} for {len(self.programs[site].keys())} programs') 

    def _calculate_time_grid(self) -> Optional[Time]:
        """Create the array with an entry for each day in the time_range, provided it exists."""
        if self.time_range is not None:
            # Add one day to make the time_range inclusive since using arange.
            return Time(np.arange(self.time_range[0].jd, self.time_range[1].jd + 1.0, (1.0 * u.day).value), format='jd')
        return None

    def _calculate_night_events(self, site: Site) -> NoReturn:
        """Load night events to collector"""
        if self.time_grid is not None:
            
            tz = TIME_ZONES[site]
            site_location = GEOGRAPHICAL_LOCATIONS[site]
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
        
        disperser = None
        camera = None
        gmos_configuration = None
        decker = None
        mask = None
        cross_disperser = None
        acquisition_mirror = None

        central_wavelength = None
        disperser_lambda = None
        wavelength = None
        
        if 'GMOS' in instrument_name:
            fpu_names = []
            fpu_widths = []
            if 'fpu' in configuration:
                fpu_list = fpu_xml_translator(configuration['fpu'])

                for fpu in fpu_list:
                    fpu_names.append(fpu['name'])
                    fpu_widths.append(fpu['width'])
            if 'customSlitWidth' in configuration:
                for custom_widths in configuration['customSlitWidth']:
                    fpu_widths.append(custom_mask_width(custom_widths))
            
            fpu_custom_mask = configuration['fpuCustomMask'] if 'fpuCustomMask' in configuration else None
            
            gmos_configuration = GMOSConfiguration(fpu_names, fpu_widths, fpu_custom_mask)

        if 'disperser' in configuration:
            disperser = [d.split('_', 1)[0] for d in configuration['disperser']]

        if 'NIRI' in instrument_name:
            camera = configuration['camera'] if 'camera' in configuration else None
            mask = configuration['mask'] if 'mask' in configuration else None

        if 'GNIRS' in instrument_name:
            cross_disperser = configuration['crossDispersed'] if 'crossDispersed' in configuration else None
            acquisition_mirror = configuration['acquisitionMirror'] if 'acquisitionMirror' in configuration else None
            decker = configuration['decker'] if 'decker' in configuration else None

        if instrument_name in ['IGRINS', 'MAROON-X']:
            disperser = ['XD']
        
        if 'Flamingos2' in instrument_name:
            decker = configuration['decker'] if 'decker' in configuration else None
        
        if 'Alopeke' in instrument_name:
            instrument_name = 'Alopeke'
        
        wavelength_config = None
        if 'centralWavelength' in configuration:
            central_wavelength = configuration['centralWavelength']
        if 'disperserLambda' in configuration:
            disperser_lambda = configuration['disperserLambda']
        if 'wavelength' in configuration:
            wavelength = configuration['wavelength']

        if central_wavelength is not None or disperser_lambda is not None or wavelength is not None:
            wavelength_config = WavelengthConfiguration(central_wavelength, disperser_lambda, wavelength)
        
        return Instrument(instrument_name, disperser,
                          gmos_configuration, camera,
                          decker, acquisition_mirror,
                          mask, cross_disperser,
                          wavelength_config)

    def _readzip(self,
                 zipfile: str,
                 xmlselect: List[str],
                 site: Site,
                 selection: List[ObservationStatus] = [ObservationStatus.ONGOING, ObservationStatus.READY],
                 tas=None):
        """ Populate Database from the zip file of an ODB backup """

        with ZipFile(zipfile, 'r') as zip:
            names = zip.namelist()
            names.sort()

            for name in names:
                if any(xs in name for xs in xmlselect):
                    tree = ElementTree.fromstring(zip.read(name))
                    program = tree.find('container')
                    (active, complete) = check_status(program)
                    if active and not complete:
                        self._process_observation_data(site, program, selection,tas)

    def _process_observation_data(self,
                                  site: Site,
                                  program_data,
                                  selection: List[ObservationStatus],
                                  tas: Dict[str, Dict[str, float]]) -> NoReturn:
        """Parse XML file to Observation objects and other data structures."""
        program_id = get_program_id(program_data)
        notes = get_program_notes(program_data)
        program_mode = get_program_mode(program_data)
        band = get_program_band(program_data)
        award = get_program_awarded_time(program_data)
        is_thesis = is_program_thesis(program_data)
        too_status = get_too_status(program_data)

        year = program_id[3:7]
        next_year = str(int(year) + 1)
        semester = program_id[7]

        # Determine the program start and end times.
        if 'FT' in program_id:
            program_start, program_end = get_ft_program_dates(notes, semester, year, next_year)
            # If still undefined, use the values from the previous observation
            if program_start is None:
                program_start = self.programs[site][-1].start
                program_end = self.programs[site][-1].end

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

        # Used time from Time Accounting Summary (tas) information
        used = tas[program_id]['prgtime'] if tas and program_id in tas else 0.0 * u.hour

        collected_program = Program(program_id,
                                    program_mode,
                                    band,
                                    is_thesis,
                                    award,
                                    used,
                                    too_status,
                                    program_start,
                                    program_end)
        self.programs[site][program_id] = collected_program

        for raw_observation, group in get_observation_info(program_data):
            classes = list(dict.fromkeys(get_obs_class(raw_observation)))
            # Precedence for choosing the obsclass from the classes list
            obsclass = select_obsclass(classes)
            status = get_obs_status(raw_observation)
            obs_odb_id = get_obs_id(raw_observation)

            if (obsclass in self.obs_classes) and (status in selection):
                logging.info(f'Adding {obs_odb_id}.')

                total_time = get_obs_time(raw_observation)

                # Target Info
                target_name, ra, dec = get_target_coords(raw_observation)

                if target_name is None:
                    target_name = 'None'
                if ra is None and dec is None:
                    ra = 0.0
                    dec = 0.0
                target_coords = SkyCoord(ra, dec, frame='icrs', unit=(u.deg, u.deg))
                target_mags = get_target_magnitudes(raw_observation, baseonly=True)
                targets = get_targets(raw_observation)
                target_designation = None
                target_tag = None
                if targets:
                    for target in targets:                        
                        try:
                            target_group_name = target['group']['name']
                            if target_group_name == 'Base':
                                target_tag = TargetTag(target['tag'])
                                if target_tag is not None and target_tag != TargetTag.Sidereal:
                                    target_designation = target['num'] if target_tag is TargetTag.MajorBody else target['des']
                        except:
                            pass

                target = Target(target_name, target_tag, target_mags, target_designation, target_coords)
                # Observation Priority
                priority = get_priority(raw_observation)
                # Instrument Configuration
                instrument_name = get_instrument(raw_observation)
                instrument_config = get_inst_configs(raw_observation)
                if 'name' in instrument_config:
                    instrument_name = instrument_config['name'][0]

                # ToO status
                too_status = get_obs_too_status(raw_observation, self.programs[site][program_id].too_status)

                # Sky Conditions
                conditions = get_conditions(raw_observation, label=False)

                if conditions is None or not conditions:
                    sky_cond = SkyConditions()
                else:
                    parse_conditions = conditions_parser(conditions)                    
                    sky_cond = SkyConditions(*parse_conditions)

                # Elevation constraints        
                elevation_type, min_elevation, max_elevation = get_elevation(raw_observation)
                elevation_type = str_to_elevation_type(elevation_type)
                min_elevation, max_elevation = str_to_float(min_elevation), str_to_float(max_elevation)
                elevation_constraints = ElevationConstraints(elevation_type, min_elevation, max_elevation)

                # Charged observation time
                # Used time from Time Accounting Summary (tas) information
                obs_time = 0
                if tas and program_id in tas and obs_odb_id in tas[program_id]:
                    obs_time = max(0, tas[program_id][obs_odb_id]['prgtime'].value)

                # Total observation time
                # for IGRINS, update tot_time to take telluric into account if there is time remaining
                calibration_time = 0
                if 'IGRINS' in instrument_name.upper() and total_time.total_seconds() / 3600. - obs_time > 0.0:
                    calibration_time = 10 / 60  # fractional hour

                inst_config = Collector._instrument_setup(instrument_config, instrument_name)

                start, duration, repeat, period = get_windows(raw_observation)
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
                        wstart -= self.observations[site][-1].acquisition()
                        windows.append(Time([round_min(wstart), round_min(wend)]))
                self.obs_windows[site].append(windows)

                # Observation number in program
                collected_program.add_observation(self.observation_num)

                # Check if group exists, add if not
                if group['key'] not in self.scheduling_groups[site].keys():
                    self.scheduling_groups[site][group['key']] = {'name': group['name'], 'idx': []}
                    collected_program.add_group(group['key'])
                self.scheduling_groups[site][group['key']]['idx'].append(Collector.observation_num)

                # Get Horizons coordinates for nonsidereal targets (write file for future use)
                # if target_tag in ['asteroid', 'comet', 'major-body']: NOTE: This does not work, and it should!

                #self.priority.append(priority)

                #logging.debug([oc  for oc in classes if oc != 'ACQ'][0] )
                self.observations[site].append(Observation(Collector.observation_num,
                                                     obs_odb_id,
                                                     band,
                                                     ObservationClass(obsclass.lower()),
                                                     obs_time,
                                                     total_time.total_seconds() / 3600. + calibration_time,
                                                     inst_config,
                                                     sky_cond,
                                                     elevation_constraints,
                                                     target,
                                                     status,
                                                     too_status))
                Collector.observation_num += 1

    def create_time_array(self):

        timesarr = []

        for i in range(len(self.time_grid)):
            tmin = min([MAX_NIGHT_EVENT_TIME] + [self.night_events[site]['twi_eve12'][i] for site in self.sites])
            tmax = max([MIN_NIGHT_EVENT_TIME] + [self.night_events[site]['twi_mor12'][i] for site in self.sites])

            tstart = round_min(tmin, up=True)
            tend = round_min(tmax, up=False)
            n = np.int((tend.jd - tstart.jd) / self.time_slot_length.to(u.day).value + 0.5)
            times = Time(np.linspace(tstart.jd, tend.jd - self.time_slot_length.to(u.day).value, n), format='jd')
            timesarr.append(times)

        return timesarr

    # TODO: This could be an static method but it should be some internal process or API call to ENV.
    # TODO: As it will need to possibly modify information in the Collector at a future point, we leave it as
    # TODO: non-static for now.
    def get_actual_conditions(self) -> Conditions:
        time_blocks = [Time(["2021-04-24 04:30:00", "2021-04-24 08:00:00"], format='iso', scale='utc')]

        # TODO: Use of these keys in the dictionary is asking for typo trouble.
        # TODO: We should have concrete types, like an enum.
        # TODO: Since this is limited to here, I'm not going to change this yet.
        variants = {
            # 'IQ20 CC50': {'iq': IQ.IQ20, 'cc': CC.CC50, 'wv': WV.WVANY, 'wdir': -1, 'ws': -1},
            'IQ70 CC50': {'iq': IQ.IQ70, 'cc': CC.CC50, 'wv': WV.WVANY, 'wdir': 330. * u.deg, 'wsep': 40. * u.deg,
                          'wspd': 5.0 * u.m / u.s, 'tb': time_blocks},
            # 'IQ70 CC70': {'iq': IQ.IQ70, 'cc': CC.CC70, 'wv': WV.WVANY, 'wdir': -1, 'wsep': 30.*u.deg, 'wspd': 0.0*u.m/u.s, 'tb': time_blocks},
            # 'IQ85 CC50': {'iq': IQ.IQ85, 'cc': CC.CC50, 'wv': WV.WVANY, 'wdir': -1, 'wsep': 30.0*u.deg, 'wspd': 0.0*u.m/u.s, 'tb': time_blocks},
            # 'IQ85 CC70': {'iq': IQ.IQ85, 'cc': CC.CC70, 'wv': WV.WVANY, 'wdir': -1, 'wsep': 30.0*u.deg, 'wspd': 0.0*u.m/u.s, 'tb': time_blocks},
            # 'IQ85 CC80': {'iq': IQ.IQANY, 'cc': CC.CC80, 'wv': WV.WVANY, 'wdir': -1, 'wsep': 30.0*u.deg, 'wspd': 0.0*u.m/u.s, 'tb': []}
        }

        selected_variant = variants['IQ70 CC50']

        sky = SkyConditions(SB.SBANY, selected_variant['cc'], selected_variant['iq'], selected_variant['wv'])
        wind = WindConditions(selected_variant['wsep'], selected_variant['wspd'], selected_variant['wdir'], time_blocks)
        return Conditions(sky, wind)
