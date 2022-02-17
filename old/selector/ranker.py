import astropy.units as u
import numpy as np
import selector.horizons as hz

from astropy.coordinates import SkyCoord
from astropy.time import Time

from typing import List
from common.structures.site import Site, GEOGRAPHICAL_LOCATIONS
from common.structures.target import TargetTag
import os


class Ranker:
    def __init__(self, sites: List[Site], times: List[Time],) -> None:
        self.sites = sites
        self.times = times
             
    # TODO: This method is duplicated and should be replace by a better Horizons API
    def _query_coordinates(self, obs, site, night, tag, des, coords,
                           ephem_dir, site_location, overwrite=False, checkephem=False, ):
        # Query coordinates, including for nonsidereal targets from Horizons
        # iobs = list of observations to process, otherwise, all non-sidereal

        query_coords = []

        for nn in night:
            tstart = self.times[nn][0].strftime('%Y%m%d_%H%M')
            tend = self.times[nn][-1].strftime('%Y%m%d_%H%M')

            if tag is TargetTag.Sidereal:
                coord = coords
            else:
                if tag is None or des is None:
                    coord = None
                else:
                    if tag is TargetTag.Comet:
                        hzname = 'DES=' + des + ';CAP'
                    elif tag == 'asteroid':
                        hzname = 'DES=' + des + ';'
                    else:
                        hzname = hz.GetHorizonId(des)

                    # File name
                    ephname = ephem_dir + '/' + site.name + '_' + des.replace(' ', '').replace('/', '') + '_' + \
                            tstart + '-' + tend + '.eph'

                    ephexist = os.path.exists(ephname)
                    
                    if checkephem and ephexist and not overwrite:
                        # Just checking that the ephem file exists, dont' need coords
                        coord = None
                    else:
                        
                        try:
                            horizons = hz.Horizons(site.value.upper(), airmass=100., daytime=True)
                            time, ra, dec = horizons.Coordinates(hzname, self.times[nn][0], 
                                                                 self.times[nn][-1], 
                                                                 step='1m',
                                                                 file=ephname, overwrite=overwrite)
                            coord = None
                        except Exception:
                            print('Horizons query failed for ' + des)
                            coord = None
                        else:
                            coord = SkyCoord(ra, dec, frame='icrs', unit=(u.rad, u.rad))

            query_coords.append(coord)
        
        return query_coords

    def score(self, visits, programs, inight,  ephem_dir, 
              pow=2, metpow=1.0, vispow=1.0, whapow=1.0, remaining=None):

        params = self._params()
        
        combine_score = lambda x: np.array([np.max(x)]) if 0 not in x else np.array([0])   
        
        for visit in visits: 
            site = visit.site
            site_location = GEOGRAPHICAL_LOCATIONS[site]
            visit_score = np.empty((0, len(self.times[inight])), dtype=float)
            for obs in [*visit.observations, *visit.calibrations]:

                score = np.zeros(len(self.times[inight]))
                program_id = obs.get_program_id()   
                program = programs[program_id]
        
                if remaining is None:
                    remaining = (obs.length - obs.observed) * u.hr

                cplt = (program.used_time + remaining) / program.time

                # Metric and slope
                metrc, metrc_s = self._metric_slope(np.array([cplt.value]),
                                                    np.ones(1, dtype=int) * program.band,
                                                    np.ones(1) * 0.8, params, 
                                                    pow=pow, thesis=program.thesis,
                                                    thesis_factor=1.1)
                # Get coordinates
                coord = self._query_coordinates(obs, site, [inight], 
                                                obs.target.tag, obs.target.designation, 
                                                obs.target.coordinates, ephem_dir,
                                                site_location)[0]
                # HA/airmass
                ha = obs.visibility.hour_angle[inight]

                if coord is not None:
                    if site_location.lat < 0. * u.deg:
                        decdiff = site_location.lat - np.max(coord.dec)
                    else:
                        decdiff = np.min(coord.dec) - site_location.lat
                else:
                    decdiff = 90. * u.deg

                if abs(decdiff) < 40. * u.deg:
                    c = np.array([3., 0.1, -0.06])  # weighted to slightly positive HA
                else:
                    c = np.array([3., 0., -0.08])  # weighted to 0 HA if Xmin > 1.3

                wha = c[0] + c[1] * ha / u.hourangle + (c[2] / u.hourangle ** 2) * ha ** 2
                kk = np.where(wha <= 0.0)[0][:]
                wha[kk] = 0.

                # p = metrc[0] * wha  # Match Sebastian's Sept 30 Gurobi test?
                # p = metrc[0] * metrc_s[0] * self.visfrac[site_name][ii] * wha  # My favorite
                # p = metrc_s[0] * self.visfrac[site_name][ii] * wha # also very good
                p = (metrc[0] ** metpow) * (obs.visibility.fraction ** vispow) * (wha ** whapow)

                score[obs.visibility.visibility[inight]] = p[obs.visibility.visibility[inight]]

                obs.score = score
                
                visit_score = np.append(visit_score, np.array([score]), axis=0)
            
            visit.score = np.apply_along_axis(combine_score, 0, visit_score)[0]


    def _params(self):
        params9 = {'1': {'m1': 1.406, 'b1': 2.0, 'm2': 0.50, 'b2': 0.5, 'xb': 0.8, 'xb0': 0.0, 'xc0': 0.0},
          '2': {'m1': 1.406, 'b1': 1.0, 'm2': 0.50, 'b2': 0.5, 'xb': 0.8, 'xb0': 0.0, 'xc0': 0.0},
          '3': {'m1': 1.406, 'b1': 0.0, 'm2': 0.50, 'b2': 0.5, 'xb': 0.8, 'xb0': 0.0, 'xc0': 0.0},
          '4': {'m1': 0.00, 'b1': 0.1, 'm2': 0.00, 'b2': 0.0, 'xb': 0.8, 'xb0': 0.0, 'xc0': 0.0},
          }
        # m2 = {'3': 0.5, '2': 3.0, '1':10.0} # use with b1*r where r=3
        m2 = {'4': 0.0, '3': 1.0, '2': 6.0, '1':20.0} # use with b1 + 5.
        xb = 0.8
        r = 3.0
        # b1 = np.array([6.0, 1.0, 0.2])
        b1 = 1.2
        for band in ['3', '2', '1']:
        #     b2 = b1*r - m2[band]
            # intercept for linear segment
            b2 = b1 + 5. - m2[band]
            # parabola coefficient so that the curves meet at xb: y = m1*xb**2 + b1 = m2*xb + b2
            m1 = (m2[band]*xb + b2)/xb**2
            params9[band]['m1'] = m1
            params9[band]['m2'] = m2[band]
            params9[band]['b1'] = b1
            params9[band]['b2'] = b2
            params9[band]['xb'] = xb
            # zeropoint for band separation
            b1 += m2[band]*1.0 + b2
        return params9

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
