from greedy_max.instrument import Instrument
from collector import Collector
from zipfile import ZipFile
import xml.etree.cElementTree as ElementTree

from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u

import sb
from selector.vskyutil import nightevents

from collector import Collector

import numpy as np

from xmlutils import *
from ranker import Ranker
from greedy_max.schedule import Observation, Visit
from greedy_max.category import Category

MAX_AIRMASS = '2.3'

class Selector:

    def __init__(self, collector: Collector, sites=None, times=None, time_range=None, ephemdir=None, dt=1.0*u.min) -> None:
        self.sites = sites                   # list of EarthLocation objects
        self.time_range = time_range         # Time object, array for visibility start/stop dates
        self.time_grid = None                # Time object, array with entry for each day in time_range
        self.dt = dt                         # time step for times

        self.ranker = Ranker(collector.obsid, time_range, sites)
        self.collector = collector
        if times is not None:
            self.times = times
            self.dt = self.times[0][1] - self.times[0][0]

    def _standard_time(self, instruments, wavelengths, modes, cal_len):
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

    def create_pool(self):
        collected_observations = self.collector.observations
        scheduling_groups = self.collector.scheduling_groups


        ## Create Visits
        visits = [] 
        for idx, group in enumerate(scheduling_groups.items()):

            obs_idxs = group['idx']
            #can_be_split = group['split']
        
            #standard_time = int(np.ceil(group['pstdt'].to(u.h).value/ dt.to(u.h).value))
            #site = Site.GS if sum(ttab_gngs['weight_gs'][grp_id]) > 0 else Site.GN

            instruments = [collected_observations[obs].instrument
                           for obs in obs_idxs]
            wavelengths = set([inst.wavelength() for inst in instruments]) 
            
            modes = [[collected_observations[obs].observation_mode()
                           for obs in obs_idxs]]
            
           
            if len(obs_idxs) > 1: #group 
                
                observations = []
                calibrations = []
                
                for obs_idx in obs_idxs:
                    observation = collected_observations[obs_idx]
                    if observation.category == Category.Science or observation.category == Category.ProgramCalibration:
                        observations.append(observation)
                    else:
                        calibrations.append(observation)

            else: #single observation 
                observation = collected_observations[obs_idxs[0]] 
               
                observations = [observation] if observation == Category.Science or Category.ProgramCalibration else []
                calibrations = [observation] if observation == Category.PartnerCalibration  else []

            can_be_split = False if len(observations) > 1 or len(calibrations) > 0 else True
            standard_time = self._standard_time(instruments,wavelengths,modes,len(calibrations))
            visits.append(Visit(idx, site, observations, calibrations, 
                                        can_be_split, standard_time))

        ## Ranker score
      
        self.ranker.visibility( )
        scored_visits = self.ranker.score()
            
