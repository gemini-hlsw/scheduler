from typing import Optional

import os
import logging
import bz2
import pandas as pd
import astropy.units as u
import numpy as np
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity

from common.minimodel import Site, Variant, CloudCover, ImageQuality, WaterVapor

logging.basicConfig(level=logging.INFO)

class Env:
   
    def __init__(self):
        self.site_data = {}
        self.site_data_by_night = {}
        self.night = {}

        for site in Site:
            site_lc = site.name.lower()
            input_filename = self.data_file_path(f'{site_lc}_weather_data.pickle.bz2')

            if(os.path.exists(input_filename)):
                logging.info(f'Reading {input_filename}')
                with bz2.open(input_filename) as input_file:
                        input_data = pd.read_pickle(input_file)
                        self.site_data[site_lc] = input_data 
                        #logging.info(self.site_data[site_lc]["Time_Stamp_UTC"].loc[150:200])
                        logging.info(f'\t{len(self.site_data[site_lc].columns)} columns, {len(self.site_data[site_lc])} rows')
                # self.site_data_by_night[site_lc] = self.night
                # for night in self.site_data[site_lc]["Time_Stamp_UTC"]:
                #     night_date = night.date()
                #     #logging.info(night_date)
                #     self.site_data_by_night[site_lc][night_date] = self.site_data[site_lc][0]
            else:
                print(f'Error, {input_filename} not found')
        
        #print(self.site_data_by_night["gn"]["2016-04-08"])

    def get_actual_conditions_variant(self,
                                      site: Site,
                                      times: Time) -> Optional[Variant]:
        """
        Return the weather variant.
        This should be site-based and time-based.
        """
        night_length = len(times)

        return Variant(
            iq=np.full(night_length, ImageQuality.IQ70),
            cc=np.full(night_length, CloudCover.CC50),
            wv=np.full(night_length, WaterVapor.WVANY),
            wind_dir=Angle(np.full(night_length, 330.0), unit='deg'),
            wind_sep=Angle(np.full(night_length, 40.0), unit='deg'),
            wind_spd=Quantity(np.full(night_length, 5.0 * u.m / u.s))
        )
    
    def data_file_path(self, filename: str) -> str:
        return os.path.join('..', '..', 'data', filename)

# test = Env()
# print(test)