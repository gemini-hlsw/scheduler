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

from greedy_max.instrument import Instrument
from greedy_max.site import Site
from greedy_max.band import Band
from greedy_max.schedule import Observation
from greedy_max.category import Category

from typing import List

#from ranker import Ranker

MAX_AIRMASS = '2.3'

   
def uniquelist(seq):
    # Make a list of unique values
    # http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def ot_timing_windows(strt, dur, rep, per, verbose=False):
    # Turn OT timing constraints into more natural units
    # Inputs are lists
    # Match output from GetWindows

    nwin = len(strt)

    # A date or duration to use for infinity (length of LP)
    infinity = 3. * 365. * 24. * u.h

    timing_windows = []
    for jj in range(nwin):

        # The timestamps are in milliseconds
        # The start time is unix time (milliseconds from 1970-01-01 00:00:00) UTC
        # Time requires unix time in seconds
        t0 = float(strt[jj]) * u.ms
        begin = Time(t0.to_value('s'), format='unix', scale='utc')

        # duration = -1 means forever
        duration = float(dur[jj])
        if duration == -1.0:
            duration = infinity
        else:
            # duration =  duration * u.ms
            duration = duration / 3600000. * u.h

        # repeat = -1 means infinite
        repeat = int(rep[jj])
        if repeat == -1:
            repeat = 1000
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

def roundMin(time, up=False):
    # Round a time down (truncate) or up to the nearest minute
    # time : astropy.Time
    # up: bool   Round up?

    t = time.copy()
    t.format = 'iso'
    t.out_subfmt = 'date_hm'
    if up:
        sec = int(t.strftime('%S'))
        if sec != 0:
            t += 1.0*u.min
    return Time(t.iso, format='iso', scale='utc')

class Collector:
    def __init__(self, sites: 
                       List[Site], 
                       semesters: List[str], 
                       program_types: List[str], 
                       obsclasses: List[str], 
                       times=None, 
                       time_range=None, 
                       dt=1.0*u.min) -> None:
        
        self.sites = sites                   # list of EarthLocation objects
        self.semesters = semesters
        self.program_types =  program_types
        self.obs_classes = obsclasses
        self.time_range = time_range         # Time object, array for visibility start/stop dates
        
        self.time_grid = self._calculate_time_grid()               # Time object, array with entry for each day in time_range
        self.dt = dt                         # time step for times

        self.observations = []
        self.programs = {}

        if times is not None:
            self.times = times
            self.dt = self.times[0][1] - self.times[0][0]

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
        #self.progstart = []
        # self.progend = []
        self.target_name = []
        self.target_tag = []
        self.target_des = []
        self.coord = []
        self.mags = []
        self.toostatus = []
        self.priority = []
        self.acqmode = []
        #self.targets = []
        #self.guidestars = []
        self.instconfig = []
        #self.pamode = []
        self.tot_time = []  # total program time
        self.obs_time = []  # used/scheduled time
        self.conditions = []
        self.elevation = []
        self.obs_windows = []
        #self.notes = []
    

    def load(self, path):
        
        #config fiLE?
        site_name = Site.GS.value #NOTE: temporary hack for just using one site
        #print(site_name)
        xmlselect = [site_name.upper() + '-' + sem + '-' + prog_type for sem in self.semesters for prog_type in self.program_types]

        #print(xmlselect)
        # Site details
        if site_name == 'gn':
            site = EarthLocation.of_site('gemini_north')
            self.timezones[Site(site_name)] = pytz.timezone(site.info.meta['timezone'])
            self.locations[Site(site_name)] = site
        elif site_name == 'gs':
            site = EarthLocation.of_site('gemini_south')
            self.timezones[Site(site_name)] = pytz.timezone(site.info.meta['timezone'])
            self.locations[Site(site_name)] = site
        else:
            print('ERROR: site_name must be "gs" or "gn".')
        
        self._calculate_night_events()
        site.info.meta['name'] = site_name
       
        sitezip = {'GN': '-0715.zip', 'GS': '-0830.zip'}
        zip_path = f"{path}/{(self.time_range[0] - 1.0*u.day).strftime('%Y%m%d')}{sitezip[site.info.meta['name'].upper()]}"
        print(zip_path)

        time_accounting = self._load_tas(path, site.info.meta['name'].upper())
        
        self._readzip(zip_path, xmlselect, site_name, tas=time_accounting, obsclasses=self.obs_classes)
    

    def _calculate_time_grid(self):

        if self.time_range is not None:
            # Add one day to make the time_range inclusive since using arange
            return Time(np.arange(self.time_range[0].jd, self.time_range[1].jd + 1.0, \
                                            (1.0*u.day).value), format='jd')
        return None

    def _calculate_night_events(self):

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


    def _load_tas(self, path, ssite):

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
        if tas:
            tas = vstack([tas, tmp])
        else:
            tas = tmp.copy()

        return sumtas_date(tas, tadate)

    def _readzip(self, zipfile, xmlselect, site, selection=['ONGOING', 'READY'], obsclasses=['SCIENCE'], tas=None):
        ''' Populate Database from the zip file of an ODB backup'''

        with ZipFile(zipfile, 'r') as zip:
            names = zip.namelist()
            names.sort()
            #print(names, xmlselect)
            #print(dir(odb))
            for name in names:
                if any(xs in name for xs in xmlselect):
                    tree = ElementTree.fromstring(zip.read(name))
                    program = tree.find('container')
                    (active, complete) = CheckStatus(program)
                    #print(name, active, complete)
                    if active and not complete:
                        self._process_observation_data(program, selection, obsclasses, tas, site)
 
    def _elevation_contrains(self, elevation_type, max_elevation, min_elevation):
        
        if elevation_type == 'NONE' or elevation_type == 'NULL':
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

    def _instrument_setup(self, configuration, instrument_name):
        instconfig = {} 

        fpuwidths = []
        disperser = 'NONE'
        for key in configuration.keys():
            ulist = uniquelist(configuration[key])
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

        #             print(instconfig)
        if any(inst in instrument_name.upper() for inst in ['IGRINS', 'MAROON-X']):
            disperser = 'XD'
        #return instconfig
        return Instrument(instrument_name,disperser,instconfig)
        #self.instconfig.append(instrument)

    #NOTE: this need to be an observation method
    def _acqtime(self, iobs):
        # Determine acquisition time in min for instrument/mode
        # Times taken from odb.GetObsTime and OT
        #print(self.instconfig, iobs)
        inst = self.instconfig[iobs]

        mode = inst.observation_mode()
        acqtime = 10.0*u.min

        gmosacq = {'imaging': 6.*u.min, 'longslit': 16.*u.min, 'ifu': 18.*u.min, 'mos': 18.*u.min}
        f2acq = {'imaging': 6.*u.min, 'longslit': 20.*u.min, 'mos': 30.*u.min}

        acquistion_lookup = {
                                'GMOS': gmosacq[mode],
                                'Flamingos2': f2acq[mode],
                                'NIFS': 11.*u.min,
                                'GNIRS':  15.*u.min,
                                'NIRI': 6.*u.min,
                                'GPI': 10.*u.min,
                                'GSAOI':  30.*u.min,
                                'Alopeke': 6.0*u.min,
                                'Zorro': 6.0*u.min,
                                'MAROON-X': 10*u.min,
                                'IGRINS': 10*u.min,
                                'Visitor Instrument': 10*u.min
        }
           
        return  acquistion_lookup['GMOS'] if 'GMOS' in inst.name else acquistion_lookup[inst.name]
   
    def _process_observation_data(self, program, selection, obsclasses, tas, site):
        
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
            award = 10 * float(award) * u.hour if unit == 'nights' else float(award) * u.hour
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
            if not program_start:

                proglist = list(self.programs.copy())
                program_start = self.programs[proglist[-1]]['progstart']
                program_end = self.programs[proglist[-1]]['progend'] 
                
        else:

            
            beginning_semester_A = Time(year + "-02-01 20:00:00", format='iso')
            end_semester_A = Time(year + "-08-01 20:00:00", format='iso')
            beginning_semester_B =  Time(next_year + "-02-01 20:00:00", format='iso')
            end_semester_B = Time(next_year + "-08-01 20:00:00", format='iso')
            # This covers 'Q', 'LP' and 'DD' program observations
            if semester == 'A':
                program_start = beginning_semester_A
                # Band 1, non-ToO, programs are 'persistent' for the following semester
                program_end = beginning_semester_B if band == Band.Band1 else end_semester_A

            else:
                program_start = end_semester_A
                program_end = end_semester_B if band == Band.Band1 else beginning_semester_B


        # Flexible boundaries - could be type-dependent
        program_start -= 14. * u.day
        program_end += 14. * u.day


        # Thesis program?
        thesis = GetThesis(program)

        # ToO status
        toostat = GetTooStatus(program)


        # Used time from Time Accounting Summary (tas) information
        if tas and program_id in tas:
            used = tas[program_id]['prgtime']
        else:
            used = 0.0 * u.hour

        self.programs[program_id] = {'mode': program_mode, 'band': band, 'progtime': award, 'usedtime': used,
                                    'thesis': thesis, 'toostatus': toostat,
                                    'progstart': program_start, 'progend': program_end, 'idx': [], 'groups': []}

        raw_observations, groups = GetObservationInfo(program)

        if not raw_observations:
            #print('Parser issue to get observation info')
            return

        #print(len(raw_observations))
        #print(len(groups))

        for raw_observation, group in zip(raw_observations,groups):

            classes = uniquelist(GetClass(raw_observation))
            status = GetObsStatus(raw_observation)
            obs_odb_id = GetObsID(raw_observation)

            if any(obs_class in obsclasses for obs_class in classes) and (status in selection):
                
                print('Adding ' + obs_odb_id, end='\r')
                total_time = GetObsTime(raw_observation)
                t, r, d = GetTargetCoords(raw_observation) #No clue what t, r or d means
                if t is None:
                    t = 'None'
                if r is None:
                    r = 0.0
                    d = 0.0
                target_mags = GetTargetMags(raw_observation, baseonly=True)
                targets = GetTargets(raw_observation)
                priority = GetPriority(raw_observation)
                instrument_name = GetInstrument(raw_observation)
                instrument_config = GetInstConfigs(raw_observation)

                if 'name' in instrument_config:
                    instrument_name = instrument_config['name'][0]
                
                too_status = GetObsTooStatus(raw_observation, self.programs[program_id]['toostatus'])
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
                                if target_tag != 'sidereal' and target_tag != '':
                                    if target_tag == 'major-body':
                                        des = target['num']
                                    else:
                                        des = target['des']
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
                self._elevation_contrains(elevation_type,max_elevation,min_elevation) 


                start, duration, repeat, period = GetWindows(raw_observation) 
                # This makes a list of timing windows between progstart and progend
                timing_windows = ot_timing_windows(start, duration, repeat, period)
                
                progstart = self.programs[program_id]['progstart']
                progend = self.programs[program_id]['progend']
                
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
                self.programs[program_id]['idx'].append(self.nobs)


                # Check if group exists, add if not
                if group['key'] not in self.scheduling_groups.keys():
                    self.scheduling_groups[group['key']] = {'name': group['name'], 'idx': []}
                    self.programs[program_id]['groups'].append(group['key'])
                self.scheduling_groups[group['key']]['idx'].append(self.nobs)

                    #print(target_tag)
                # Get Horizons coordinates for nonsidereal targets (write file for future use)
                #if target_tag in ['asteroid', 'comet', 'major-body']: NOTE: This does not work, and it should!
                #print(group)
                #print(self.scheduling_groups)

                
                #self.obsid.append(obs_odb_id)
                #self.band.append(band)
                self.obstatus.append(status)
                #self.obsclass.append(classes[0])
                self.target_name.append(t)
                self.target_tag.append(target_tag)
                self.target_des.append(des)
                self.coord.append(SkyCoord(r, d, frame='icrs', unit=(u.deg, u.deg)))
                self.mags.append(target_mags)
                self.toostatus.append(toostat.lower())
                self.priority.append(priority)
                #self.obs_time.append(obs_time)
                #self.tot_time.append(total_time.total_seconds() / 3600. + calibration_time) 
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
        nnight = len(self.time_grid)

        for ii in range(nnight):
            tmin = Time('2200-01-01 00:00:00', format='iso', scale='utc')
            tmax = Time('1980-01-01 00:00:00', format='iso', scale='utc')
            for site in self.sites:
                #site_name = site.info.meta['name']
                if self.night_events[site]['twi_eve12'][ii] < tmin:
                    tmin = self.night_events[site]['twi_eve12'][ii]
                if self.night_events[site]['twi_mor12'][ii] > tmax:
                    tmax = self.night_events[site]['twi_mor12'][ii]

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