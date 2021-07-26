from selector.vskyutil import nightevents
from zipfile import ZipFile
import xml.etree.cElementTree as ElementTree

from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

import sb

import pytz
import os      
import calendar

from xmlutils import *
from ranker import Ranker
from greedy_max.band import Band
from greedy_max.schedule import Observation, Visit
from greedy_max.site import Site

MAX_AIRMASS = '2.3'

class Selector:

    def __init__(self, sites=None, times=None, time_range=None, ephemdir=None, dt=1.0*u.min) -> None:
        self.sites = sites                   # list of EarthLocation objects
        self.time_range = time_range         # Time object, array for visibility start/stop dates
        self.time_grid = None                # Time object, array with entry for each day in time_range
        self.dt = dt                         # time step for times

        if times is not None:
            self.times = times
            self.dt = self.times[0][1] - self.times[0][0]

        self.ephemdir = ephemdir
        self.night_events = {}
        self.programs = {}
        self.schedgroups = {}
        self.instconfig = []
        self.nobs = 0

        self.programs = {}
        self.obsid = []
        self.obstatus = []
        self.obsclass = []
        self.nobs = 0
        #self.band = []
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
        self.ivisarr = {}
        self.vishours = {}
        self.visfrac = {}
        self.modestats = {}
        self.ha = {}
        self.targalt = {}
        self.targaz = {}
        self.targparang = {}
        self.airmass = {}
        self.sbcond = {}
        self.schedgroups = {}

        self.already_load = False
    
    def select_visits(self):
        print("mm")

        if not self.already_load:
            self._load()
        
        ranker = Ranker(self.obsid, self.time, self.sites)

        ranker.visibility(self.ephemdir, self.target_des, self.target_tag, 
                          self.coord, self.conditions, self.obs_windows, self.elevation, 
                          self.night_events)
        
    def _create_visits(self, site, inight, params, pow=2.0, metpow=1.0, vispow=1.0, whapow=1.0, verbose=False ):
    
        # Process and update scheduling groups

        site_name = site.info.meta['name']

        # For stats
        grp_tot_times = []
        grp_sci_times = []
        grp_nsci = []  # number of science/prog_cal observations in group

        for prgid in list(self.programs.copy().keys()):
            
            for group in self.programs[prgid]['groups']:
                
                #print(self.schedgroups[group]['name'])
                #self.schedgroups[group]['sci_time'] = 0.0  # Totel SCI exec time
                #self.schedgroups[group]['tot_time'] = 0.0  # Total exec time
                #self.schedgroups[group]['obs_time'] = 0.0  # Total time observed
                iptr = []  # indices of partner_cals
                ipmax = -1  # index of longest parter_cal
                grpinst = []
                grpwav = []
                grpmode = []
                grpcond = {'iq': 1.0, 'cc': 1.0, 'wv': 1.0}
                split = True

                scores = np.empty((0, len(self.times[inight])), dtype=float)
                comb_score = np.zeros(len(self.times[inight]), dtype=float)
                #     print('shape scores:', scores.shape)
                nsciobs = 0
                for idx in self.schedgroups[group]['idx']:
                    
                    # Information needed to apply calibration rules
                    # Wavelength [um]
                    if self.instconfig[idx]['inst'] not in grpinst:
                        grpinst.append(self.instconfig[idx]['inst'])
                    # GNIRS
                    # TODO: homogenize the key name and unit [microns] for wavelength
                    if 'centralWavelength' in self.instconfig[idx].keys():
                        for wav in self.instconfig[idx]['centralWavelength']:
                            if float(wav) not in grpwav:
                                grpwav.append(float(wav))
                    # GMOS
                    elif 'disperserLambda' in self.instconfig[idx].keys():
                        for wav in self.instconfig[idx]['disperserLambda']:
                            if float(wav) / 1000. not in grpwav:
                                grpwav.append(float(wav) / 1000.)
                    # Visitor instruments
                    elif 'wavelength' in self.instconfig[idx].keys():
                        for wav in self.instconfig[idx]['wavelength']:
                            if float(wav) not in grpwav:
                                grpwav.append(float(wav))
                    # Mode
                    mode = self.obsmode(idx)
                    if mode not in grpmode:
                        grpmode.append(mode)

                    # First pass group analysis
                    if self.obsclass[idx] in ['SCIENCE', 'PROG_CAL']:
                        nsciobs += 1
                        self.schedgroups[group]['tot_time'] += self.tot_time[idx]
                        self.schedgroups[group]['obs_time'] += self.obs_time[idx]
                        if self.obsclass[idx] in ['SCIENCE']:
                            # Remaining science exec time, need to check that the new acq is included
                            # This assumes all the same inst/mode, may need to separate by those
                            self.schedgroups[group]['sci_time'] += (self.tot_time[idx] - \
                                                                        self.obs_time[idx]) * u.hr
                            # Conditions
                            # Use the most restrictive value for each condition
                            # BG handled with visibility calculations
                            for cond in list(grpcond.copy().keys()):
                                if self.conditions[idx][cond] < grpcond[cond]:
                                    grpcond[cond] = self.conditions[idx][cond]

                    elif self.obsclass[idx] in ['PARTNER_CAL']:
                        iptr.append(idx)
                        # Initialize ipmax
                        if ipmax == -1:
                            ipmax = idx
                        # Identify telluric? (NIR spec)
                        # Find index of the longest partner cal, this may be used to set the group duration
                        if self.tot_time[idx] > self.tot_time[ipmax]:
                            ipmax = idx

                            # Save partner cal info, so don't have to do it again
                self.schedgroups[group]['iptr'] = iptr

                # Can the group be split?
                if nsciobs > 1 or len(iptr) > 0:
                    split = False
                self.schedgroups[group]['split'] = split

                # Second pass to calculate score
                print(self.schedgroups[group]['sci_time'])
                sci_time_metric = self.schedgroups[group]['sci_time']
                # limit the time used for calculating the metric if sci_time longer than a night.
                if (self.schedgroups[group]['sci_time']) > \
                        self.night_events[site_name]['night_length'][inight]:
                    sci_time_metric = 3.0 * u.h

                for idx in self.schedgroups[group]['idx']:
                    if self.obsclass[idx] in ['SCIENCE', 'PROG_CAL']:
                        # Score
                        s = self.calc_score(site, inight, idx, params, pow=pow, metpow=metpow, vispow=vispow, \
                                            whapow=whapow, remaining=sci_time_metric)
                        #                 print(s.shape)
                        scores = np.append(scores, np.array([s]), axis=0)
                #                 print('shape scores:', scores.shape)

                #     Store combined score
                #         print(scores.shape, len(scores))
                if scores.shape[0] > 0:
                    comb_score = combine_scores(scores)
                    
                self.schedgroups[group]['score'] = comb_score
        
                # Store conditions
                self.schedgroups[group]['cond'] = grpcond

                # Instrument list
                self.schedgroups[group]['inst'] = grpinst

                # Mode
                self.schedgroups[group]['mode'] = grpmode

                # Wavelengths
                self.schedgroups[group]['wavlen'] = grpwav

                # Evaluate partner cals (e.g multiple tellurics)
                ttel = 0.0 * u.hr

                if len(iptr) == 1:
                    self.schedgroups[group]['tot_time'] += self.tot_time[iptr[0]]
                    self.schedgroups[group]['obs_time'] += self.obs_time[iptr[0]]
                elif len(iptr) > 1:
                    # if NIR spectroscopy, use time for one or two tellurics based on wavelength
                    # wav < 2.5um, every 1.5 hr, >2.5 um every 1 hr
                    # https://www.gemini.edu/observing/resources/near-ir-resources
                    #
                    # if NIR imaging, one phot standard every 2 hours
                    if any(item in grpinst for item in ['Flamingos2', 'GNIRS', 'NIFS', 'IGRINS']):
                        if any(item in grpmode for item in ['longslit', 'ifu', 'xd', 'mos']):
                            if all(item <= 2.5 for item in grpwav):
                                ttel = 1.5 * u.hr
                            else:
                                ttel = 1.0 * u.hr
                        elif 'imaging' in grpmode:
                            ttel = 2.0 * u.hr

                        # Number of standards needed
                        nstd = max(1, int(self.schedgroups[group]['sci_time'] // ttel))
                        if nstd == 1:
                            # If use one, pick the longest observation
                            self.schedgroups[group]['tot_time'] += self.tot_time[ipmax]
                            self.schedgroups[group]['obs_time'] += self.obs_time[ipmax]
                        else:
                            # If use both, sum all
                            for ii in iptr:
                                self.schedgroups[group]['tot_time'] += self.tot_time[ii]
                                self.schedgroups[group]['obs_time'] += self.obs_time[ii]

                self.schedgroups[group]['pstd_time'] = ttel

                grp_tot_times.append(self.schedgroups[group]['tot_time'])
                grp_sci_times.append(self.schedgroups[group]['sci_time'])

        return

    def _load(self, path, time_range, time_accounting):
        
        #config fiLE?
        xmlselect = []
        obs_classes = ['SCIENCE', 'PROG_CAL', 'PARTNER_CAL']
        prog_types = ['Q', 'LP', 'FT', 'DD']
        site_name = 'gs'
        semesters = ['2018B','2019A']
        xmlselect = [site_name.upper() + '-' + sem + '-' + prog_type for sem in semesters for prog_type in prog_types]

        # Site details
        if site_name == 'gn':
            site = EarthLocation.of_site('gemini_north')
            hst = pytz.timezone(site.info.meta['timezone'])
        elif site_name == 'gs':
            site = EarthLocation.of_site('gemini_south')
            clt = pytz.timezone(site.info.meta['timezone'])
        else:
            print('ERROR: site_name must be "gs" or "gn".')
        site.info.meta['name'] = site_name
        utc = pytz.timezone('UTC')

        ephemdir = path + '/ephem/'
        if not os.path.exists(ephemdir):
            os.makedirs(ephemdir)
        sitezip = {'GN': '-0715.zip', 'GS': '-0830.zip'}
        zip_path = f"{path}/{(time_range[0] - 1.0*u.day).strftime('%Y%m%d')}{sitezip[site.info.meta['name'].upper()]}"
        print(zip_path)
        self._readzip(zip_path, xmlselect, tas=time_accounting, obsclasses=obs_classes, 
                        sites=[site], time_range = time_range, ephemdir=ephemdir)
        self.already_load = True
    
    def _readzip(self,zipfile, xmlselect, selection=['ONGOING', 'READY'], obsclasses=['SCIENCE'],
            tas=None, database=None, odbplan=None):
        ''' Populate Database from the zip file of an ODB backup'''

        with ZipFile(zipfile, 'r') as zip:
            names = zip.namelist()
            names.sort()
            print(names, xmlselect)
            #print(dir(odb))
            for name in names:
                if any(xs in name for xs in xmlselect):
                    tree = ElementTree.fromstring(zip.read(name))
                    program = tree.find('container')
                    (active, complete) = CheckStatus(program)
                    #print(name, active, complete)
                    if active and not complete:
                        self._process_observation(program, selection, obsclasses, tas)
    
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

                if verbose:
                    print(start.iso, end.iso)

                timing_windows.append(Time([start, end]))

        return timing_windows
    
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
        instconfig = {'inst': instrument_name} #can be a class ? too OOP?
        for key in configuration.keys():
            ulist = self.uniquelist(configuration[key])
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
                    fpuwidths = []
                    fpunames = []
                    for fpu in fpulist:
                        fpunames.append(fpu['name'])
                        fpuwidths.append(fpu['width'])
                    ulist = fpunames
                    instconfig['fpuWidth'] = fpuwidths
                if key == 'customSlitWidth':
                    fpuwidths = []
                    for cwidth in ulist:
                        fpuwidths.append(CustomMaskWidth(cwidth))
                    instconfig['fpuWidth'] = fpuwidths
            instconfig[key] = ulist
        #             print(instconfig)
        if any(inst in instrument_name.upper() for inst in ['IGRINS', 'MAROON-X']):
            instconfig['disperser'] = ['XD']
        #return instconfig
        self.instconfig.append(instconfig)

    def _process_observation(self, program, selection, obsclasses, tas):
        
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
            award = 10 * float(award) * u.hour if unit == 'nights' else award = float(award) * u.hour
        else:
            award = 0.0 * u.hour
        
        year = program_id[3:7]
        yp = str(int(year) + 1)
        semester = program_id[7]
        
        if 'FT' in program_id:
           program_start, program_end = GetFTProgramDates(notes,semester,year,yp) 
           
           # If still undefined, use the values from the previous observation
           if not program_start:
                proglist = list(self.programs.copy())
            #             print(prgid, proglist)
                program_start = self.programs[proglist[-1]]['progstart']
                program_end = self.programs[proglist[-1]]['progend'] 
        else:
            
            beginning_semester_A = Time(year + "-02-01 20:00:00", format='iso')
            beginning_semester_B = Time(year + "-08-01 20:00:00", format='iso')
            # This covers 'Q', 'LP' and 'DD' program observations
            if semester == 'A':
                progam_start = beginning_semester_A
                # Band 1, non-ToO, programs are 'persistent' for the following semester
                program_end = beginning_semester_A if band == Band.Band1 else beginning_semester_B
            else:
                program_start = beginning_semester_B
                program_end = beginning_semester_B if band == Band.Band1 else beginning_semester_A


        # Flexible boundaries - could be type-dependent
        progam_start -= 14. * u.day
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

        raw_observation, group = GetObservationInfo(program)

        if not raw_observation:
            print('Parser issue to get observation info')
            return

        classes = self.uniquelist(GetClass(raw_observation))
        status = GetObsStatus(raw_observation)
        obs_odb_id = GetObsID(raw_observation)

        if any(obs_class in obsclasses for obs_class in classes) and (status in selection):
            
            total_time = GetObsTime(raw_observation)
            t, r, d = GetTargetCoords(raw_observation) #No clue what t, r or d means
            target_mags = GetTargetMags(raw_observation, baseonly=True)
            targets = GetTargets(raw_observation)
            priority = GetPriority(raw_observation)
            instrument = GetInstrument(raw_observation)
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
        
        start, duration, repeat, period = GetWindows(raw_observation)       

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
            
        calibration_time = 0


        # Total observation time
        # for IGRINS, update tot_time to take telluric into account if there is time remaining
        calibration_time = 0
        if 'IGRINS' in instrument_name.upper() and total_time.total_seconds() / 3600. - obs_time > 0.0:
            calibration_time = 10/ 60   # fractional hour

        self._instrument_setup(instrument_config,instrument_name)

        # Conditions
        cond = conditions.split(',')
        condf = sb.convertcond(cond[0], cond[1], cond[2], cond[3])

        # Elevation constraints
        self._elevation_contrains(elevation_type,max_elevation,min_elevation) 

         # This makes a list of timing windows between progstart and progend
        timing_windows = self.ot_timing_windows(start, duration, repeat, period)
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
                wstart -= self.acqtime(-1)
                windows.append(Time([roundMin(wstart), roundMin(wend)]))
        
        self.programs[program_id]['idx'].append(self.nobs)


        # Check if group exists, add if not
        if group['key'] not in self.schedgroups.keys():
            self.schedgroups[group['key']] = {'name': group['name'], 'idx': []}
            self.programs[program_id]['groups'].append(group['key'])
        self.schedgroups[group['key']]['idx'].append(self.nobs)

            #print(target_tag)
        # Get Horizons coordinates for nonsidereal targets (write file for future use)
        #if target_tag in ['asteroid', 'comet', 'major-body']: NOTE: This does not work, and it should!
            
        self.nobs += 1
        self.obsid.append(obs_odb_id)
        self.obstatus.append(status)
        self.obsclass.append(classes[0])
        self.target_name.append(t)
        self.coord.append(SkyCoord(r, d, frame='icrs', unit=(u.deg, u.deg)))
        self.mags.append(target_mags)
        self.toostatus.append(toostat.lower())
        self.priority.append(priority)
        self.obs_time.append(obs_time)
        self.tot_time.append(total_time.total_seconds() / 3600. + calibration_time) 
        self.conditions.append({'iq': condf[0], 'cc': condf[1], 'bg': condf[2], 'wv': condf[3]})
        
    
