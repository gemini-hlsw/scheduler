import logging

from collector import Collector
import collector.vskyutil as vs
import collector.sb as sb

from common.structures.conditions import Conditions, SkyConditions
from common.structures.elevation import ElevationType
from common.structures.observation_status import ObservationStatus
from common.structures.target import TargetTag
from common.structures.too_type import ToOType

from selector.visibility import Visibility
from selector.ranker import Ranker
import selector.horizons as hz

from greedy_max.schedule import Observation, Visit
from greedy_max.category import Category
from common.structures.site import Site

from resource_mock.resources import Resources

from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

import multiprocessing
import numpy as np
from tqdm import tqdm

from typing import List, NoReturn, Dict, Set, Iterable


class Selector:

    def __init__(self, collector: Collector, sites=None, times=None, time_range=None, dt=1.0 * u.min) -> None:
        self.sites = sites  # list of EarthLocation objects
        self.time_range = time_range  # Time object, array for visibility start/stop dates
        self.dt = dt  # time step for times

        self.collector = collector
        if times is not None:
            self.times = times
            self.dt = self.times[0][1] - self.times[0][0]
        else:
            self.times = collector.create_time_array()

        self.selection = {}

    @staticmethod
    def _standard_time(instruments: List[str], wavelengths: Set[float], modes: List[List[str]], cal_len: int) -> u.hr:
        standard_time = 0.0 * u.hr
        if cal_len > 1:
            if any(item in instruments for item in ['Flamingos2', 'GNIRS', 'NIFS', 'IGRINS']):
                if all(item <= 2.5 for item in wavelengths):
                    standard_time = 1.5 * u.hr
                else:
                    standard_time = 1.0 * u.hr
            elif 'imaging' in modes:
                standard_time = 2.0 * u.hr
        return standard_time

    def _get_airmass(self, altit: Angle) -> np.ndarray:
        """true airmass for an altitude.
        Equivalent of getAirmass in the QPT, based on vskyutil.true_airmass
        https://github.com/gemini-hlsw/ocs/blob/12a0999bc8bb598220ddbccbdbab5aa1e601ebdd/bundle/edu.gemini.qpt.client/src/main/java/edu/gemini/qpt/core/util/ImprovedSkyCalcMethods.java#L119

        Based on a fit to Kitt Peak airmass tables, C. M. Snell & A. M. Heiser, 1968,
        PASP, 80, 336.  Valid to about airmass 12, and beyond that just returns
        secz minus 1.5, which won't be quite right.

        Parameters
        ----------

        altit : Angle, float or numpy array
            Altitude above horizon.

        Returns : float
        """
        # takes an Angle and return the true airmass, based on
        # 	 a tabulation of the mean KPNO
        #            atmosphere given by C. M. Snell & A. M. Heiser, 1968,
        # 	   PASP, 80, 336.  They tabulated the airmass at 5 degr
        #            intervals from z = 60 to 85 degrees; I fit the data with
        #            a fourth order poly for (secz - airmass) as a function of
        #            (secz - 1) using the IRAF curfit routine, then adjusted the
        #            zeroth order term to force (secz - airmass) to zero at
        #            z = 0.  The poly fit is very close to the tabulated points
        # 	   (largest difference is 3.2e-4) and appears smooth.
        #            This 85-degree point is at secz = 11.47, so for secz > 12
        #            just return secz   */

        #    coefs = [2.879465E-3,  3.033104E-3, 1.351167E-3, -4.716679E-5]

        altit = np.asarray(altit.to_value(u.rad).data)
        scalar_input = False
        if altit.ndim == 0:
            altit = altit[None]  # Makes 1D
            scalar_input = True

        # ret = np.zeros(len(altit))
        ret = np.full(len(altit), 58.)
        ii = np.where(altit > 0.0)[0][:]
        if len(ii) != 0:
            ret[ii] = 1. / np.sin(altit[ii])  # sec z = 1/sin (altit)

        kk = np.where(np.logical_and(ret >= 0.0, ret < 12.))[0][:]
        if len(kk) != 0:
            seczmin1 = ret[kk] - 1.
            coefs = np.array([-4.716679E-5, 1.351167E-3, 3.033104E-3, 2.879465E-3, 0.])
            ret[kk] = ret[kk] - np.polyval(coefs, seczmin1)
            # print "poly gives",  np.polyval(coefs,seczmin1)

        if scalar_input:
            return np.squeeze(ret)
        return ret

    def _calculate_visibility(self, site, des, tag, coo, conditions, elevation, obs_windows, times, lst,
                              sunalt, moonpos, moondist, moonalt, sunmoonang, location, ephem_dir, sbtwo=True,
                              overwrite=False, extras=True):

        nt = len(times)
        if tag is not TargetTag.Sidereal:
            if tag is None or des is None:
                coord = None
            else:
                usite = site.value.upper()
                horizons = hz.Horizons(usite, airmass=100., daytime=True)

                if tag is TargetTag.Comet:
                    hzname = 'DES=' + des + ';CAP'
                elif tag is TargetTag.Asteroid:
                    hzname = 'DES=' + des + ';'
                else:
                    hzname = hz.GetHorizonId(des)

                # File name
                ephname = usite + '_' + des.replace(' ', '').replace('/', '') + '_' + \
                          times[0].strftime('%Y%m%d_%H%M') + '-' + times[-1].strftime('%Y%m%d_%H%M') + '.eph'

                try:
                    time, ra, dec = horizons.Coordinates(hzname, times[0], times[-1], step='1m', \
                                                         file=ephem_dir + '/' + ephname, overwrite=overwrite)
                except Exception:
                    print('Horizons query failed for ' + des)
                    coord = None
                else:
                    coord = SkyCoord(ra, dec, frame='icrs', unit=(u.rad, u.rad))
        else:
            coord = coo

        ivis = np.array([], dtype=int)
        ha = Angle(np.full(nt, 12.), unit=u.hourangle)
        targalt = Angle(np.zeros(nt), unit=u.radian)
        targaz = Angle(np.zeros(nt), unit=u.radian)
        targparang = Angle(np.zeros(nt), unit=u.radian)
        airmass = np.zeros(nt)
        sbcond = np.ones(nt)
        if coord is not None:
            ha = lst - coord.ra
            ha.wrap_at(12. * u.hour, inplace=True)
            targalt, targaz, targparang = vs.altazparang(coord.dec, ha, location.lat)
            # airmass = vs.xair(90. * u.deg - targalt)
            airmass = self._get_airmass(targalt)
            targprop = airmass if elevation.elevation_type is ElevationType.AIRMASS else ha.value

            # Sky brightness
            if conditions.sb < 1.0:
                targmoonang = coord.separation(moonpos)
                if sbtwo:
                    # New algorithm
                    skyb = sb.sb2(180. * u.deg - sunmoonang, targmoonang, moondist, 90. * u.deg - moonalt,
                                  90. * u.deg - targalt, 90. * u.deg - sunalt)
                else:
                    # Use current QPT algorithm
                    skyb = sb.sb(180. * u.deg - sunmoonang, targmoonang, 90. * u.deg - moonalt,
                                 90. * u.deg - targalt, 90. * u.deg - sunalt)
                sbcond = sb.sb_to_cond(skyb)

            # Select where sky brightness and elevation constraints are met
            # Evenutally want to allow some observations, e.g. calibration, in twilight
            ix = np.where(np.logical_and(sbcond <= conditions.sb,
                                         np.logical_and(sunalt <= -12. * u.deg,
                                                        np.logical_and(
                                                            targprop >= elevation.min_elevation,
                                                            targprop <= elevation.max_elevation))))[0]

            # Timing window constraints
            #     iit = np.array([], dtype=int)
            for constraint in obs_windows:
                iit = np.where(np.logical_and(times[ix] >= constraint[0],
                                              times[ix] <= constraint[1]))[0]
                ivis = np.append(ivis, ix[iit])

        return ivis, ha, targalt, targaz, targparang, airmass, sbcond if extras else ivis

    @staticmethod
    def _check_instrument_availability(resources: Resources,
                                       site: Site,
                                       instruments_of_obs: Iterable[str]) -> bool:
        return all(resources.is_instrument_available(site, instrument) for instrument in instruments_of_obs)

    @staticmethod
    def _check_conditions(visit_sky_conditions: SkyConditions,
                          actual_sky_conditions: SkyConditions) -> bool:
        return (visit_sky_conditions.iq >= actual_sky_conditions.iq and
                visit_sky_conditions.cc >= actual_sky_conditions.cc and
                visit_sky_conditions.wv >= actual_sky_conditions.wv)

    # TODO: Maybe we should move this out of the Selector and into the conditions file since it is really a
    # TODO: check on SkyConditions and uses no data from the Selector.
    @staticmethod
    def _match_conditions(visit_conditions: SkyConditions,
                          actual_conditions: SkyConditions,
                          neg_ha: bool,
                          too_status: ToOType) -> np.ndarray:

        skyiq = actual_conditions.iq.value
        skycc = actual_conditions.cc.value
        skywv = actual_conditions.wv.value

        skyiq = np.asarray(skyiq)
        skycc = np.asarray(skycc)
        skywv = np.asarray(skywv)
        scalar_input = False
        if skyiq.ndim == 0:
            scalar_input = True
            skyiq = skyiq[None]
        if skycc.ndim == 0:
            skycc = skycc[None]
        if skywv.ndim == 0:
            skywv = skywv[None]

        cmatch = np.ones(len(skyiq))

        # Where actual conditions worse than requirements
        bad_iq = skyiq > visit_conditions.iq
        bad_cc = skycc > visit_conditions.cc
        bad_wv = skywv > visit_conditions.wv

        # Multiply weights by 0 where actual conditions worse than required.
        i_bad_cond = np.where(np.logical_or(np.logical_or(bad_iq, bad_cc), bad_wv))[0][:]
        cmatch[i_bad_cond] = 0

        # Multiply weights by skyiq/iq where iq better than required and target
        # does not set soon and not a rapid ToO.
        i_better_iq = np.where(skyiq < visit_conditions.iq)[0][:]
        if len(i_better_iq) != 0 and neg_ha and too_status != ToOType.RAPID:
            cmatch[i_better_iq] = cmatch[i_better_iq] * skyiq / visit_conditions.iq.value
        # cmatch[i_better_iq] = cmatch[i_better_iq] * (1.0 - (iq - skyiq))

        i_better_cc = np.where(skycc < visit_conditions.cc)[0][:]
        if len(i_better_cc) != 0 and neg_ha and too_status != ToOType.RAPID:
            cmatch[i_better_cc] = cmatch[i_better_cc] * skycc / visit_conditions.cc.value
        if scalar_input:
            cmatch = np.squeeze(cmatch)

        return cmatch

    # TODO: What is an 'sbtwo'?
    def visibility(self, site: Site, jobs=0, ephem_path=None, overwrite=False, sbtwo=True) -> NoReturn:
        """
        Main driver to calculate the visibility for each observation 
        """
        obs_windows = self.collector.obs_windows
        timezones = self.collector.timezones
        site_location = self.collector.locations[site]

        # NOTE: observation set indifferent of site, this might (should?) change
        # if the query for observations is site bound, for example to manage OR groups
        observations = self.collector.observations

        night_length = len(self.times)

        time_slot_length = (self.times[0][1] - self.times[0][0]).to_value('hour') * u.hr

        # TODO: This was 1.e5. I think it should be 1e-5, so I changed it, but confirm.
        # TODO: Also, should we raise a RunTimeException here instead of printing and returning?
        if abs((time_slot_length.to(u.min) - 1.0 * u.min).value) > 1e-5:
            logging.error(f'Time slot length must be 1 min, but is {time_slot_length.to(u.min)}.')
            return

        # TODO: This is not being used due to the commented out code below.
        # Use all available CPU cores if unspecified, but no more than the maximum.
        jobs = min(multiprocessing.cpu_count() if jobs < 1 else jobs, multiprocessing.cpu_count())

        obsvishours = {site: np.zeros((len(observations), night_length))}

        num_observations = len(observations)
        visibilities = [[]] * num_observations
        hour_angles = [[]] * num_observations
        target_alts = [[]] * num_observations
        target_azs = [[]] * num_observations
        target_parangs = [[]] * num_observations
        airmass = [[]] * num_observations
        sky_brightness = [[]] * num_observations

        for period in range(night_length):
            sun_position = vs.lpsun(self.times[period])
            lst = vs.lpsidereal(self.times[period], site_location)
            sunalt, sunaz, sunparang = vs.altazparang(sun_position.dec, lst - sun_position.ra, site_location.lat)

            moonpos, moondist = vs.accumoon(self.times[period], site_location)
            moonalt, moonaz, moonparang = vs.altazparang(moonpos.dec, lst - moonpos.ra, site_location.lat)

            if sbtwo:
                sunmoonang = sun_position.separation(moonpos)  # for sb2
            else:
                midnight = vs.local_midnight_Time(self.times[period][0], timezones[site])
                moonmid, moondistmid = vs.accumoon(midnight, site)
                sunmid = vs.lpsun(midnight)
                sunmoonang = sunmid.separation(moonmid)  # for sb

            '''
            res = Parallel(n_jobs=jobs)(delayed(self._calculate_visibility)(site, target_des[id],
                                                    target_tag[id], coord[id],
                                                    conditions[id], elevation[id],
                                                    obs_windows[id], self.times[period], 
                                                    lst, sunalt, #times should be ii
                                                    moonpos, moondist, 
                                                    moonalt, sunmoonang, 
                                                    site_location, ephem_dir,
                                                    sbtwo=sbtwo, overwrite=overwrite, extras=True)  
                                                    for id in tqdm(range(len(observations))))
            
            '''
            res = []

            for obs in tqdm(observations):
                res.append(self._calculate_visibility(site,
                                                      obs.target.designation,
                                                      obs.target.tag,
                                                      obs.target.coordinates,
                                                      obs.sky_conditions,
                                                      obs.elevation,
                                                      obs_windows[obs.idx],
                                                      self.times[period],
                                                      lst, sunalt,
                                                      moonpos, moondist,
                                                      moonalt, sunmoonang,
                                                      site_location, ephem_path,
                                                      sbtwo=sbtwo, overwrite=overwrite, extras=True))

            if period == 0:
                for obs in observations:
                    obsvishours[site][obs.idx, period] = len(res[obs.idx][0]) * time_slot_length.value
                    visibilities[obs.idx].append(res[obs.idx][0])
                    hour_angles[obs.idx].append(res[obs.idx][1])
                    target_alts[obs.idx].append(res[obs.idx][2])
                    target_azs[obs.idx].append(res[obs.idx][3])
                    target_parangs[obs.idx].append(res[obs.idx][4])
                    airmass[obs.idx].append(res[obs.idx][5])
                    sky_brightness[obs.idx].append(res[obs.idx][6])
                break

        for obs in observations:
            obs_id = obs.idx
            sum_obsvishr = np.sum(obsvishours[site][obs_id, :])

            # visfrac needs to be calculated per night, for the current night through
            # the end of the current period
            if sum_obsvishr > 0.0:
                visfrac = (obs.length - obs.observed) / sum_obsvishr
            else:
                visfrac = 0.0

            obs.visibility = Visibility(visibilities[obs_id],
                                        obsvishours[site][obs_id, :],
                                        hour_angles[obs_id],
                                        visfrac,
                                        target_alts[obs_id],
                                        target_azs[obs_id],
                                        target_parangs[obs_id],
                                        airmass[obs_id],
                                        sky_brightness[obs_id])

        logging.info('Done calculating observation visibility.')

    def create_pool(self) -> Dict[Site, List[Visit]]:
        """
        Process to create the Visits for each site base on the information from Collector
        """
        collected_observations = self.collector.observations
        scheduling_groups = self.collector.scheduling_groups
        visits = {site: [] for site in self.sites}

        for site in self.sites:
            # Create Visits
            for idx, group in enumerate(scheduling_groups.values()):
                obs_idxs = group['idx']

                instruments = [collected_observations[obs].instrument
                               for obs in obs_idxs]

                wavelengths = set([wav for inst in instruments for wav in inst.wavelength()])

                modes = [[collected_observations[obs].instrument.observation_mode()
                          for obs in obs_idxs]]

                if len(obs_idxs) > 1:  # group
                    observations = []
                    calibrations = []

                    for obs_idx in obs_idxs:
                        observation = collected_observations[obs_idx]
                        if (observation.category == Category.Science
                                or observation.category == Category.ProgramCalibration):
                            observations.append(observation)
                        else:
                            calibrations.append(observation)

                else:  # single observation
                    observation = collected_observations[obs_idxs[0]]

                    observations = [observation] if (observation == Category.Science
                                                     or Category.ProgramCalibration) else []
                    calibrations = [observation] if observation == Category.PartnerCalibration else []

                can_be_split = len(observations) <= 1 and len(calibrations) == 0

                standard_time = Selector._standard_time(instruments, wavelengths, modes, len(calibrations))
                visits[site].append(Visit(idx, site, observations, calibrations,
                                          can_be_split, standard_time))

        return visits

    def select(self, visits: List[Visit],
               inight: int,
               site: Site,
               actual_conditions: Conditions,
               resources: Resources,
               ephem_dir: str) -> List[Visit]:
        """
        Select the visits that are possible to be scheduled under current conditions from the pool.
        Return a collection of visits for each site.
        """
        ranker = Ranker(self.sites, self.times)

        ranker.score(visits,
                     self.collector.programs,
                     self.collector.locations,
                     inight,
                     ephem_dir)

        actual_sky_conditions = actual_conditions.sky
        actual_wind_conditions = actual_conditions.wind

        selected = []
        for visit in visits:

            if visit.length() - visit.observed() > 0:
                negative_hour_angle = True
                dispersers_in_obs = []
                fpus_in_obs = []
                instruments_in_obs = []
                valid_in_obs = []
                status_of_obs = []
                vishours_of_obs = []
                too_status = 'none'
                visit_conditions = visit.sky_conditions

                # Check observations can be selected for the visit
                for obs in visit.observations:

                    # If HA < 0 in first time step, then we don't consider it setting at the start
                    if (negative_hour_angle and obs.category in ['science', 'prog_cal'] and
                            obs.visibility.hour_angle[0].value > 0):
                        negative_hour_angle = False

                    # Check wind constraints
                    if actual_wind_conditions.wind_direction != -1 and actual_wind_conditions.wind_speed > 0 * u.m / u.s:
                        wind_conditions = actual_wind_conditions.get_wind_conditions(obs.visibility.azimuth[inight])

                    # NOTE: first condition always going to be true. Why is needed?
                    if too_status != ToOType.RAPID and obs.too_status != ToOType.NONE:
                        too_status = obs.too_status

                    # Check for correct instrument configuration and check comp in other sites. 
                    # NOTE: This could be done before the selecting process 
                    comp_val, comp_instrument = Selector.has_complementary_mode(obs, visit.site)
                    instruments_in_obs.append(comp_instrument)
                    valid_in_obs.append(comp_val)
                    vishours_of_obs.append(obs.visibility.hours[inight])

                    if 'GMOS' in comp_instrument:
                        comp_disperser = obs.instrument.disperser
                        dispersers_in_obs.append(comp_disperser)
                        fpu = obs.instrument.configuration['fpu']
                        fpus = obs.instrument.configuration['fpuCustomMask'] if 'CUSTOM_MASK' in fpu else fpu
                        fpus_in_obs.extend(fpus)

                    status_of_obs.append(obs.status)

                instruments_in_obs = dict.fromkeys(instruments_in_obs)
                dispersers_in_obs = dict.fromkeys(dispersers_in_obs)
                fpus_in_obs = dict.fromkeys(fpus_in_obs)
                status_of_obs = dict.fromkeys(status_of_obs)

                if (all(valid_in_obs) and all(hours > 0 for hours in vishours_of_obs) and
                        Selector._check_instrument_availability(resources, site, instruments_in_obs) and
                        all(status in [ObservationStatus.ONGOING, ObservationStatus.READY, ObservationStatus.OBSERVED]
                            for status in status_of_obs) and
                        Selector._check_conditions(visit_conditions, actual_sky_conditions)):

                    # CHECK FOR GMOS IF COMPONENTS ARE INSTALLED
                    if any('GMOS' in instrument for instrument in instruments_in_obs):
                        can_be_selected = False
                        for disperser in dispersers_in_obs:
                            if resources.is_disperser_available(site, disperser):
                                for fpu in fpus_in_obs:
                                    if resources.is_mask_available(site, fpu):
                                        can_be_selected = True

                        if can_be_selected:
                            # TODO: Update the scores. This could be done by the ranker?
                            match = self._match_conditions(visit_conditions,
                                                           actual_sky_conditions,
                                                           negative_hour_angle,
                                                           too_status)

                            # TODO: wind_conditions may not be initialized by this point.
                            visit.score = wind_conditions * visit.score * match

                    else:
                        selected.append(visit)

        self.selection[site] = selected
        return selected

    def selection_summary(self) -> NoReturn:
        """
        Show a summary of the selection of Visits
        """
        if not self.selection:
            print('No Visits were selected')
            return
        for site in self.sites:
            print(f'Site {site.name}')
            for visit in self.selection[site]:
                print(visit)

    @staticmethod
    def has_complementary_mode(obs: Observation, site: Site) -> tuple[bool, str]:
        """
        Determines if an observation configuration is valid for site.
        This is mainly to determine if it can be observed at an alternative site
        """
        go = False
        altinst = 'None'

        obs_site = site
        mode = obs.instrument.observation_mode()
        instrument = obs.instrument
        if obs_site != site:
            if site == Site.GN:
                if (instrument.name == 'GMOS-S' and
                        mode in ['imaging', 'longslit', 'ifu']):
                    go = True
                    altinst = 'GMOS-N'
                elif instrument.name == 'Flamingos2':
                    if mode in ['imaging']:
                        go = True
                        altinst = 'NIRI'
                    elif (mode in ['longslit'] and
                          'GCAL' not in instrument.configuration['title'] and
                          'R3000' in instrument.disperser):
                        go = True
                        altinst = 'GNIRS'
            elif site == Site.GN:

                if (instrument.name == 'GMOS-N' and
                        mode in ['imaging', 'longslit', 'ifu']):
                    go = True
                    altinst = 'GMOS-S'
                elif (instrument.name == 'NIRI' and
                      instrument.configuration['camera'] == 'F6'):
                    go = True
                    altinst = 'Flamingos2'
                    if mode in ['imaging']:
                        go = True
                        altinst = 'Flamingos2'
                    elif mode in ['longslit']:
                        if (instrument.disperser == 'D_10' and
                                'SHORT' in instrument.configuration['camera']):
                            go = True
                            altinst = 'Flamingos2'
                        elif (instrument.disperser == 'D_10' and
                              'LONG' in instrument.configuration['camera']):
                            go = True
                            altinst = 'Flamingos2'
        else:
            go = True
            altinst = obs.instrument.name

        return go, altinst
