import astropy.units as u
import multiprocessing
import numpy as np

import vskyutil as vs
import sb
import horizons as hz
import pytz
from joblib import Parallel, delayed

from astropy.coordinates import SkyCoord, Angle

from typing import List
from greedy_max.site import Site
class Ranker:
    def __init__(self, n_observations: int, times, sites: List[Site]) -> None:
        self.n_observations = n_observations
        self.times = times
        self.sites = sites
        
        
        self.ivisarr = {}
        self.vishours = {}
        self.visfrac = {}
        for site in self.sites:
            self.ivisarr[site] = []
            self.vishours[site] = []
            self.visfrac[site] = []
            self.ha[site] = []
            self.targalt[site] = []
            self.targaz[site] = []
            self.targparang[site] = []
            self.sbcond[site] = []
            self.airmass[site] = []
            
            for i in range(n_observations):
                self.ivisarr[site].append([])
                self.ha[site].append([])
                self.targalt[site].append([])
                self.targaz[site].append([])
                self.targparang[site].append([])
                self.airmass[site].append([])
                self.sbcond[site].append([])
            
        

    def visibility(self, ephemdir, target_des, target_tag, coord, conditions, 
                   obs_windows, elevation, night_events, total_times, observed_times,
                   jobs=0, overwrite=False, sbtwo= True):
        
        night_length = len(self.times)
    

        dt = (self.times[0][1] - self.times[0][0]).to_value('hour') * u.hr
        if abs((dt.to(u.min) - 1.0*u.min).value) > 1.e5:
            print('Timestep dt must be 1 min, is ', dt.to(u.min))
            return

        # How many CPU cores?
        if jobs < 1:
            jobs = multiprocessing.cpu_count()
        
        for site in self.sites:
            
            obsvishours = np.zeros((self.n_observations, night_length))

            for position, period in enumerate(self.times):
                sun_position = vs.lpsun(period)
                lst = vs.lpsidereal(period, site)
                sunalt, sunaz, sunparang = vs.altazparang(sun_position.dec, lst - sun_position.ra, site.lat)
            #         moonmid, moondistmid = vs.accumoon(midnight[ii], site)
                moonpos, moondist = vs.accumoon(period, site)
                moonalt, moonaz, moonparang = vs.altazparang(moonpos.dec, lst - moonpos.ra, site.lat)
                sunmoonang = sun_position.separation(moonpos)
                if sbtwo:
                    sunmoonang = sun_position.separation(moonpos)  # for sb2
                else:
                    midnight = vs.local_midnight_Time(period[0], pytz.timezone(site.info.meta['timezone']))
                    moonmid, moondistmid = vs.accumoon(midnight, site)
                    sunmid = vs.lpsun(midnight)
                    sunmoonang = sunmid.separation(moonmid) # for sb

                
                res = Parallel(n_jobs=jobs)(delayed(self._calculate_visibility)(site, ephemdir, target_des[id],
                                target_tag[id], coord[id],
                                conditions[id], elevation[id],
                                obs_windows[id], self.times[id], lst, sunalt,
                                moonpos, moondist, moonalt, sunmoonang,
                                night_events[site]['twi_eve12'][position],
                                night_events[site]['twi_mor12'][position],
                                sbtwo=sbtwo, overwrite=overwrite, extras=True) for id in range(self.n_observations))
            

                for id in range(self.n_observations):
                   
                    self.ivisarr[site][id].append(res[id][0])
                    obsvishours[id, position] = len(res[id][0]) * dt.value
                    if position == 0:   # just save info for the first night, uncomment to save every night
                        self.ha[site][id].append(res[id][1])
                        self.targalt[site][id].append(res[id][2])
                        self.targaz[site][id].append(res[id][3])
                        self.targparang[site][id].append(res[id][4])
                        self.airmass[site][id].append(res[id][5])
                        self.sbcond[site][id].append(res[id][6])


            for id in range(self.n_observations):
                self.vishours[site].append(obsvishours[id, :])
                sum_obsvishr = np.sum(obsvishours[id, :])
                # visfrac needs to be calculated per night, for the current night through
                # the end of the current period
                if sum_obsvishr > 0.0:
                    visfrac = (total_times[id] - observed_times[id]) / sum_obsvishr
                else:
                    visfrac = 0.0
                self.visfrac[site].append(visfrac)

        print('Done')
        return
    
    def _calculate_visibility(self, site, ephdir, des, tag, coo, conditions, elevation, obs_windows, times, lst,
           sunalt, moonpos, moondist, moonalt, sunmoonang, twi_eve12, twi_mor12, 
           sbtwo=True, overwrite=False, extras=True):

        nt = len(times)

        if tag != 'sidereal':
            if tag in ['undef', ''] or des in ['undef', '']:
                coord = None
            else:
                usite = site.info.meta['name'].upper()
                horizons = hz.Horizons(usite, airmass=100., daytime=True)

                if tag == 'comet':
                    hzname = 'DES=' + des + ';CAP'
                elif tag == 'asteroid':
                    hzname = 'DES=' + des + ';'
                else:
                    hzname = hz.GetHorizonId(des)

                # File name
                ephname = usite + '_' + des.replace(' ', '').replace('/', '') + '_' + \
                        times[0].strftime('%Y%m%d_%H%M') + '-' + times[-1].strftime('%Y%m%d_%H%M') + '.eph'

                try:
                    time, ra, dec = horizons.Coordinates(hzname, times[0], times[-1], step='1m', \
                                                        file=ephdir + '/' + ephname, overwrite=overwrite)
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
            targalt, targaz, targparang = vs.altazparang(coord.dec, ha, site.lat)
            # airmass = vs.xair(90. * u.deg - targalt)
            airmass = self._get_airmass(targalt)
            if elevation['type'] == 'AIRMASS':
                # targalt, targaz, targparang = vs.altazparang(coord.dec, ha, site.lat)
                # targprop = vs.xair(90. * u.deg - targalt)
                targprop = airmass
            else:
                targprop = ha.value
            #         ha = site.target_hour_angle(times, target).hour
            #         targprop = np.where(ha > 12., ha - 24., ha)
            #         targprop = np.where(targprop < -12., targprop + 24., targprop)

            # Sky brightness
            if conditions['bg'] < 1.0:
                targmoonang = coord.separation(moonpos)
                if sbtwo:
                    # New algorithm
                    skyb = sb.sb2(180. * u.deg - sunmoonang, targmoonang, moondist, 90. * u.deg - moonalt,
                            90. * u.deg - targalt, 90. * u.deg - sunalt)
                else:
                    # Use current QPT algorithm
                    skyb = sb.sb(180.*u.deg - sunmoonang, targmoonang, 90.*u.deg - moonalt,
                                90.*u.deg-targalt, 90.*u.deg - sunalt)
                sbcond = sb.sb_to_cond(skyb)

            # Select where sky brightness and elevation constraints are met
            # Evenutally want to allow some observations, e.g. calibration, in twilight
            ix = np.where(np.logical_and(sbcond <= conditions['bg'], np.logical_and(sunalt <= -12. * u.deg,
                        np.logical_and(targprop >= elevation['min'], targprop <= elevation['max']))))[0]

            # Timing window constraints
            #     iit = np.array([], dtype=int)
            for constraint in obs_windows:
                iit = np.where(np.logical_and(times[ix] >= constraint[0],
                                            times[ix] <= constraint[1]))[0]
                ivis = np.append(ivis, ix[iit])

        if extras:
            return ivis, ha, targalt, targaz, targparang, airmass, sbcond
        else:
            return ivis

    def _get_airmass(self,altit):
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
    
    def score(self, site, inight, iobs, params, pow=2, metpow=1.0, vispow=1.0, whapow=1.0, remaining=None):

        site_name = site.info.meta['name']

        score = np.zeros(len(self.times[inight]))

        prgid = progid(self.obsid[iobs])
        #     print(prgid)

        if remaining is None:
            remaining = (self.tot_time[iobs] - self.obs_time[iobs]) * u.hr

        cplt = (self.programs[prgid]['usedtime'] + remaining) / self.programs[prgid]['progtime']
        # Metric and slope
        metrc, metrc_s = self._metric_slope(np.array([cplt.value]),
                                      np.ones(1, dtype=int) * self.programs[prgid]['band'],
                                      np.ones(1) * 0.8, params, pow=pow, thesis=self.programs[prgid]['thesis'],
                                      thesis_factor=1.1)

        # Get coordinates
        coord = self.getcoord(iobs=iobs, inight=inight)[site_name][0][0]
        #     print(coord.ra)

        # HA/airmass
        ha = self.ha[site_name][iobs][inight]
        # xair = self.airmass[site_name][iobs][inight]

        if coord is not None:
            if site.lat < 0. * u.deg:
                decdiff = site.lat - np.max(coord.dec)
            else:
                decdiff = np.min(coord.dec) - site.lat
        else:
            decdiff = 90. * u.deg

        if abs(decdiff) < 40. * u.deg:
            c = np.array([3., 0.1, -0.06])  # weighted to slightly positive HA
        else:
            c = np.array([3., 0., -0.08])  # weighted to 0 HA if Xmin > 1.3

        wha = c[0] + c[1] * ha / u.hourangle + (c[2] / u.hourangle ** 2) * ha ** 2
        kk = np.where(wha <= 0.0)[0][:]
        wha[kk] = 0.
        #     print('wha:', wha)

        #                     p = metrc[0] * wha  # Match Sebastian's Sept 30 Gurobi test?
        #                     p = metrc[0] * metrc_s[0] * self.visfrac[site_name][ii] * wha  # My favorite
        #                     p = metrc_s[0] * self.visfrac[site_name][ii] * wha # also very good
        p = (metrc[0] ** metpow) * (self.visfrac[site_name][iobs] ** vispow) * (wha ** whapow)
        #     print('p:', p)
        #                 print(len(wha), len(p), len(score), len(self.ivisarr[site_name][ii][iobs]))
        #     score[0,self.ivisarr[site_name][iobs][inight]] = p[self.ivisarr[site_name][iobs][inight]]
        score[self.ivisarr[site_name][iobs][inight]] = p[self.ivisarr[site_name][iobs][inight]]

        #     print('score:', score.shape)

        return score

    def _metric_slope(self,completion, band, b3min, params, pow=1, thesis=False, thesis_factor=0.0): 
    
        """
        Compute the metric and the slope as a function of completness fraction and band

        Parameters
            completion: array/list of program completion fractions
            band: integer array of bands for each program
            b3min: array of Band 3 minimum time fractions (Band 3 minimum time / Allocated program time)
            params: dictionary of parameters for the metric
            pow: exponent on completion, pow=1 is linear, pow=2 is parabolic
        """

        eps = 1.e-7
        nn = len(completion)
        metric = np.zeros(nn)
        metric_slope = np.zeros(nn)
        for ii in range(nn):
            sband = str(band[ii])

            # If Band 3, then the Band 3 min fraction is used for xb
            if band[ii] == 3:
                xb = b3min[ii]
                # b2 = xb * (params[sband]['m1'] - params[sband]['m2']) + params[sband]['xb0']
            else:
                xb = params[sband]['xb']
                # b2 = params[sband]['b2']

            # Determine the intercept for the second piece (b2) so that the functions are continuous
            if pow == 1:
                b2 = xb * (params[sband]['m1'] - params[sband]['m2']) + params[sband]['xb0'] + params[sband]['b1']
            elif pow == 2:
                b2 = params[sband]['b2'] + params[sband]['xb0'] + params[sband]['b1']
                # print(sband, xb, b2)

            # Finally, calculate piecewise the metric and slope
            if completion[ii] <= eps:
                metric[ii] = 0.0
                metric_slope[ii] = 0.0
            elif completion[ii] < xb:
                metric[ii] = params[sband]['m1'] * completion[ii] ** pow + params[sband]['b1']
                metric_slope[ii] = pow * params[sband]['m1'] * completion[ii] ** (pow - 1.)
            elif completion[ii] < 1.0:
                metric[ii] = params[sband]['m2'] * completion[ii] + b2
                metric_slope[ii] = params[sband]['m2']
            else:
                metric[ii] = params[sband]['m2'] * 1.0 + b2 + params[sband]['xc0']
                metric_slope[ii] = params[sband]['m2']
                # print(metric[ii])

        if thesis:
            metric += thesis_factor
            # metric *= thesis_factor
            # metric_slope *= thesis_factor

        return metric, metric_slope
    

    def combine_scores(scores):
        # scores: 2D nd array of scores, e.g. np.array([s1, s2, s3, ...])
        # output the maximum non-zero value at each x position in vectors
        if scores.shape[0] > 1:
            vlen = scores.shape[1]
            score = np.zeros(vlen)
            for ii in range(vlen):
                vmin = np.amin(scores[:,ii])
                vmax = np.amax(scores[:,ii])
                if vmin != 0.0:
                    score[ii] = vmax
        else:
            score = scores
        return score