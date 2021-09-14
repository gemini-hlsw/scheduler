from time import time
from zipfile import ZipFile
import xml.etree.cElementTree as ElementTree

from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.table import Table, vstack

import pytz
import os      
import calendar
import numpy as np

import collector.sb as sb
from collector.vskyutil import nightevents
from collector.xmlutils import *
from collector.get_tadata import get_report, get_tas, sumtas_date
from collector.conditions import SkyConditions, WindConditions
from collector.program import Program

from greedy_max.instrument import Instrument
from greedy_max.site import Site
from greedy_max.band import Band
from greedy_max.schedule import Observation
from greedy_max.category import Category

from typing import List, Dict, Optional, NoReturn

MAX_AIRMASS = '2.3'
FUZZY_BOUNDARY = 14
CLASSICAL_NIGHT_LEN = 10
INFINITE_DURATION = 3. * 365. * 24. * u.h # A date or duration to use for infinity (length of LP)
INFINITE_REPEATS = 1000 # number to depict infinity for repeats in OT Timing windows calculations
MIN_NIGHT_EVENT_TIME = Time('1980-01-01 00:00:00', format='iso', scale='utc')
MAX_NIGHT_EVENT_TIME = Time('2200-01-01 00:00:00', format='iso', scale='utc')

def ot_timing_windows(strt, dur, rep, per, verbose=False):
    """
    Turn OT timing constraints into more natural units
    Inputs are lists
    Match output from GetWindows

    """

    timing_windows = []
    for (jj, (strt, dur)) in enumerate(zip(strt, dur)):

        # The timestamps are in milliseconds
        # The start time is unix time (milliseconds from 1970-01-01 00:00:00) UTC
        # Time requires unix time in seconds
        t0 = float(strt) * u.ms
        begin = Time(t0.to_value('s'), format='unix', scale='utc')

        # duration = -1 means forever
        duration = float(dur)
        duration = INFINITE_DURATION if duration == -1.0 else duration / 3600000. * u.h

        # repeat = -1 means infinite
        repeat = int(rep[jj])
        if repeat == -1:
            repeat = INFINITE_REPEATS
        if repeat == 0:
            repeat = 1

        # period between repeats
        # period = float(values[3]) * u.ms
        period = float(per[jj]) / 3600000. * u.h

        for ii in range(repeat):
            start = begin + float(ii) * period
            end = start + duration

            timing_windows.append(Time([start, end]))

    return timing_windows

def roundMin(time: Time, up=False) -> Time:
    """
    Round a time down (truncate) or up to the nearest minute
    time : astropy.Time
    up: bool   Round up?s
    """
    

    t = time.copy()
    t.format = 'iso'
    t.out_subfmt = 'date_hm'
    if up:
        sec = int(t.strftime('%S'))
        if sec != 0:
            t += 1.0*u.min
    return Time(t.iso, format='iso', scale='utc')

class Collector:
    def __init__(self, sites: List[Site], 
                       semesters: List[str], 
                       program_types: List[str], 
                       obsclasses: List[str], 
                       time_range=None, 
                       dt=1.0*u.min) -> None:
        
        self.sites = sites                   
        self.semesters = semesters
        self.program_types =  program_types
        self.obs_classes = obsclasses
        self.time_range = time_range         # Time object, array for visibility start/stop dates
        
        self.time_grid = self._calculate_time_grid()  # Time object, array with entry for each day in time_range
        self.dt = dt                         # time step for times

        self.observations = []
        self.programs = {}

        self.night_events = {}        
        self.scheduling_groups = {}
        self.instconfig = []
        self.nobs = 0
        
        # NOTE: This are used to used the EarthLocation and Timezone objects for functions that used those kind of 
        # objects. This can either be include in the Site class if the use of this libraries is justified. 
        self.timezones = {}
        self.locations = {}

        self.programs = {}
        self.obsid = []
        self.obstatus = []
        self.obsclass = []
        self.nobs = 0
        self.band = []

        self.target_name = []
        self.target_tag = []
        self.target_des = []
        self.coord = []
        self.mags = []
        self.toostatus = []
        self.priority = []
        self.acqmode = []
        self.instconfig = []
        self.tot_time = []  # total program time
        self.obs_time = []  # used/scheduled time
        self.conditions = []
        self.elevation = []
        self.obs_windows = []

    def load(self, path: str) -> NoReturn:
        """ Main collector method. It setups the collecting process and parameters """ 

        #config fiLE?
        site_name = Site.GS.value #NOTE: temporary hack for just using one site

        xmlselect = [site_name.upper() + '-' + sem + '-' + prog_type for sem in self.semesters for prog_type in self.program_types]

        # Site details
        if site_name == 'gn':
            site = EarthLocation.of_site('gemini_north')
        elif site_name == 'gs':
            site = EarthLocation.of_site('gemini_south')
            
        else:
            raise RuntimeError('ERROR: site_name must be "gs" or "gn".')
            
        
        self.timezones[Site(site_name)] = pytz.timezone(site.info.meta['timezone'])
        self.locations[Site(site_name)] = site

        self._calculate_night_events()
        
        sitezip = {'GN': '-0715.zip', 'GS': '-0830.zip'}
        zip_path = f"{path}/{(self.time_range[0] - 1.0*u.day).strftime('%Y%m%d')}{sitezip[site_name.upper()]}"
        print(zip_path)

        time_accounting = self._load_tas(path, site_name.upper())        
        self._readzip(zip_path, xmlselect, site_name, tas=time_accounting, obsclasses=self.obs_classes)
    

    def _calculate_time_grid(self) -> Optional[Time]:

        if self.time_range is not None:
            # Add one day to make the time_range inclusive since using arange
            return Time(np.arange(self.time_range[0].jd, self.time_range[1].jd + 1.0, \
                                            (1.0*u.day).value), format='jd')
        return None

    def _calculate_night_events(self) -> NoReturn:
        """ Load night events to collector """

        if self.time_grid is not None:

            for site in self.sites:
                tz = self.timezones[site]
                site_location = self.locations[site]
                mid, sset, srise, twi_eve18, twi_mor18, twi_eve12, twi_mor12, mrise, mset, smangs, moonillum = \
                    nightevents(self.time_grid, site_location, tz, verbose=False)
                night_length = (twi_mor12 - twi_eve12).to_value('h') * u.h
                self.night_events[site] = {'midnight': mid, 'sunset': sset, 'sunrise': srise, \
                                                            'twi_eve18': twi_eve18, 'twi_mor18': twi_mor18, \
                                                            'twi_eve12': twi_eve12, 'twi_mor12': twi_mor12, \
                                                            'night_length': night_length, \
                                                            'moonrise': mrise, 'moonset': mset, \
                                                            'sunmoonang': smangs, 'moonillum': moonillum}


    def _load_tas(self, path: str, ssite: str) -> Dict[str,Dict[str,float]]:
        """ Load Time Accouting Summary """

        date = self.time_range[0].strftime('%Y%m%d')
        plandir = path + '/nightplans/' + date + '/'
        print(plandir)    
        if not os.path.exists(plandir):
            os.makedirs(plandir)    
        tas = Table()
        tadate = (self.time_range[0] - 1.0*u.day).strftime('%Y%m%d')
        
        for sem in self.semesters:
            tafile = 'tas_' + ssite + '_' + sem + '.txt'
            if not os.path.exists(plandir + tafile):
                get_report(ssite, tafile, plandir)
                
            tmp = get_tas(plandir + tafile)
            tas = vstack([tas, tmp])

        return sumtas_date(tas, tadate)

    def _elevation_constraints(self, elevation_type, max_elevation, min_elevation):
        """ Calculate elevation constrains """

        if elevation_type == 'NONE':
            elevation_type = 'AIRMASS'
        if min_elevation == 'NULL' or min_elevation == '0.0':
            if elevation_type == 'AIRMASS':
                min_elevation = '1.0'
            else:
                min_elevation = '-5.0'
        if max_elevation == 'NULL' or max_elevation == '0.0':
            if elevation_type == 'AIRMASS':
                max_elevation = MAX_AIRMASS
            else:
                max_elevation = '5.0'
        self.elevation.append({'type': elevation_type, 'min': float(min_elevation), 'max': float(max_elevation)})

    def _instrument_setup(self, configuration: Dict[str, List[str]], instrument_name: str) -> Instrument:
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


        if any(inst in instrument_name.upper() for inst in ['IGRINS', 'MAROON-X']):
            disperser = 'XD'
       
        return Instrument(instrument_name,disperser,instconfig)
        
    def _readzip(self, 
                 zipfile: str, 
                 xmlselect: List[str], 
                 site: str,
                 selection=['ONGOING', 'READY'], 
                 obsclasses=['SCIENCE'], 
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
                        self._process_observation_data(program, selection, obsclasses, tas, site)
   
    def _process_observation_data(self, 
                                  program, 
                                  selection: List[str], 
                                  obsclasses: List[str], 
                                  tas: Dict[str,Dict[str,float]], 
                                  site: str) -> NoReturn:
        """ Parse XML file to Observation objects and other data structures """

        program_id = GetProgramID(program)
        notes = GetProgNotes(program)
        program_mode = GetMode(program)
        
        xml_band = GetBand(program)
        if xml_band == 'UNKNOWN':
            band = Band(1) if program_mode == 'CLASSICAL' else Band(0)
        else:
            band = Band(int(xml_band))

        award, unit = GetAwardedTime(program)
        if award and unit:
            award = CLASSICAL_NIGHT_LEN * float(award) * u.hour if unit == 'nights' else float(award) * u.hour
        else:
            award = 0.0 * u.hour
        
        year = program_id[3:7]
        next_year = str(int(year) + 1)
        semester = program_id[7]

        program_start = None
        program_end = None

        if 'FT' in program_id:
            program_start, program_end = GetFTProgramDates(notes,semester,year, next_year) 
            # If still undefined, use the values from the previous observation
            if program_start is None:

                proglist = self.program.copy()
                program_start = proglist[-1].start
                program_end = proglist[-1].end
                
        else:
            
            beginning_semester_1 = Time(year + "-02-01 20:00:00", format='iso')
            end_semester_1 = Time(year + "-08-01 20:00:00", format='iso')
            beginning_semester_2 =  Time(next_year + "-02-01 20:00:00", format='iso')
            end_semester_2 = Time(next_year + "-08-01 20:00:00", format='iso')
            # This covers 'Q', 'LP' and 'DD' program observations
            if semester == 'A':
                program_start = beginning_semester_1
                # Band 1, non-ToO, programs are 'persistent' for the following semester
                program_end = beginning_semester_2 if band == Band.Band1 else end_semester_1

            else:
                program_start = end_semester_1
                program_end = end_semester_2 if band == Band.Band1 else beginning_semester_2


        # Flexible boundaries - could be type-dependent
        program_start -= FUZZY_BOUNDARY * u.day
        program_end += FUZZY_BOUNDARY * u.day


        # Thesis program?
        thesis = GetThesis(program)

        # ToO status
        toostat = GetTooStatus(program)


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
        raw_observations, groups = GetObservationInfo(program)

        if raw_observations is None:
            raise RuntimeError('Parser issue to get observation info')

        for raw_observation, group in zip(raw_observations,groups):

            classes = list(dict.fromkeys(GetClass(raw_observation)))
            status = GetObsStatus(raw_observation)
            obs_odb_id = GetObsID(raw_observation)

            if any(obs_class in obsclasses for obs_class in classes) and (status in selection):
                
                print('Adding ' + obs_odb_id, end='\r')
                total_time = GetObsTime(raw_observation)
                target_name, ra, dec = GetTargetCoords(raw_observation)
                
                if target_name is None:
                    target_name = 'None'
                if ra is None and dec is None:
                    ra = 0.0
                    dec = 0.0
                
                target_mags = GetTargetMags(raw_observation, baseonly=True)
                targets = GetTargets(raw_observation)
                priority = GetPriority(raw_observation)
                instrument_name = GetInstrument(raw_observation)
                instrument_config = GetInstConfigs(raw_observation)

                #print(instrument_config)
                if 'name' in instrument_config:
                    instrument_name = instrument_config['name'][0]
                too_status = GetObsTooStatus(raw_observation, collected_program.too_status)
                conditions = GetConditions(raw_observation, label=False)

                if conditions == '' or conditions is None:
                    conditions = 'ANY,ANY,ANY,ANY'
                
                try:
                    elevation_type, min_elevation, max_elevation = GetElevation(raw_observation)
                except:
                # print('GetElevation failed for ' + o)
                    elevation_type = 'AIRMASS'
                    min_elevation = '1.0'
                    max_elevation = MAX_AIRMASS
            
            
                acquisiton_mode = 'normal'
                target_tag = 'undef'
                des = 'undef'

                if targets:
                    for target in targets:
                        try:
                            if target['group']['name'] == 'Base':
                                target_tag = target['tag']
                                if target_tag and  target_tag != 'sidereal':
                                    des = target['num'] if target_tag == 'major-body' else target['des']

                            if target['group']['name'] == 'User' and target['type'] == 'blindOffset':
                                acquisiton_mode = 'blindOffset'
                        except:
                            pass
                
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

                inst_config = self._instrument_setup(instrument_config,instrument_name)
                # Conditions
                cond = conditions.split(',')
                condf = sb.convertcond(cond[0], cond[1], cond[2], cond[3])
                sky_cond = SkyConditions(condf[0],condf[2],condf[1],condf[3])

                # Elevation constraints
                self._elevation_constraints(elevation_type,max_elevation,min_elevation) 

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
                        windows.append(Time([roundMin(wstart), roundMin(wend)]))
                self.obs_windows.append(windows)
                
                # Observation number in program
                collected_program.add(self.nobs)


                # Check if group exists, add if not
                if group['key'] not in self.scheduling_groups.keys():
                    self.scheduling_groups[group['key']] = {'name': group['name'], 'idx': []}
                    collected_program.add(group['key'])
                self.scheduling_groups[group['key']]['idx'].append(self.nobs)

                # Get Horizons coordinates for nonsidereal targets (write file for future use)
                #if target_tag in ['asteroid', 'comet', 'major-body']: NOTE: This does not work, and it should!
                
                self.obstatus.append(status)
                self.target_name.append(target_name)
                self.target_tag.append(target_tag)
                self.target_des.append(des)
                self.coord.append(SkyCoord(ra, dec, frame='icrs', unit=(u.deg, u.deg)))
                self.mags.append(target_mags)
                self.toostatus.append(toostat.lower())
                self.priority.append(priority)
                self.conditions.append({'iq': condf[0], 'cc': condf[1], 'bg': condf[2], 'wv': condf[3]})

                self.observations.append(Observation(self.nobs,
                                                    obs_odb_id,
                                                    band, 
                                                    Category(classes[0].lower()), 
                                                    obs_time, 
                                                    total_time.total_seconds() / 3600. + calibration_time,
                                                    inst_config,
                                                    sky_cond,
                                                    status,
                                                    too_status.lower()))
                self.nobs += 1

    def create_time_array(self):

        timesarr = []

        for i in range(len(self.time_grid)):
            
            tmin = min([MAX_NIGHT_EVENT_TIME] + [self.night_events[site]['twi_eve12'][i] for site in self.sites])
            tmax = max([MIN_NIGHT_EVENT_TIME] + [self.night_events[site]['twi_mor12'][i] for site in self.sites])

            tstart = roundMin(tmin, up=True)
            tend = roundMin(tmax, up=False)
            n = np.int((tend.jd - tstart.jd) / self.dt.to(u.day).value + 0.5)
            times = Time(np.linspace(tstart.jd, tend.jd - self.dt.to(u.day).value, n), format='jd')
            timesarr.append(times)

        return timesarr
    
    def get_actual_conditions(self):
        # TODO: This could be an static method but it should be some internal process or API call to ENV
        # but right now is mostly hardcoded so the plan would be as if 

        actcond = {}
        time_blocks = [Time(["2021-04-24 04:30:00", "2021-04-24 08:00:00"], format='iso', scale='utc')] #
        variants = {
            #             'IQ20 CC50': {'iq': 0.2, 'cc': 0.5, 'wv': 1.0, 'wd': -1, 'ws': -1},
            'IQ70 CC50': {'iq': 0.7, 'cc': 0.5, 'wv': 1.0, 'wdir': 330.*u.deg, 'wsep': 40.*u.deg, 'wspd': 5.0*u.m/u.s, 'tb': time_blocks},
            #             'IQ70 CC70': {'iq': 0.7, 'cc': 0.7, 'wv': 1.0, 'wdir': -1, 'wsep': 30.*u.deg, 'wspd': 0.0*u.m/u.s, 'tb': time_blocks}, 
            #             'IQ85 CC50': {'iq': 0.85, 'cc': 0.5, 'wv': 1.0, 'wdir': -1, 'wsep': 30.0*u.deg, 'wspd': 0.0*u.m/u.s, 'tb': time_blocks}, 
            #             'IQ85 CC70': {'iq': 0.85, 'cc': 0.7, 'wv': 1.0, 'wdir': -1, 'wsep': 30.0*u.deg, 'wspd': 0.0*u.m/u.s, 'tb': time_blocks},
            #             'IQ85 CC80': {'iq': 1.0, 'cc': 0.8, 'wv': 1.0, 'wdir': -1, 'wsep': 30.0*u.deg, 'wspd': 0.0*u.m/u.s, 'tb': []}
            }

        selected_variant = variants['IQ70 CC50']
        
        actcond['sky'] = SkyConditions(selected_variant['iq'], None, selected_variant['cc'],
                             selected_variant['wv'])
        actcond['wind'] = WindConditions( selected_variant['wsep'], selected_variant['wspd'],
                             selected_variant['wdir'],time_blocks)
        
        return actcond