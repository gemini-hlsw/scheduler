from typing import Optional
from datetime import date, datetime, timedelta

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


class Env:
   
    def __init__(self):
        """
        Stores weather data into a dictionary 
        that can be indexed by site and date.
        """
        self.site_data = {}
        self.site_data_by_night = {}

        for site in Site:
            site_lc = site.name.lower()
            input_filename = Env.data_file_path(f'{site_lc}_weather_data.pickle.bz2')

            if(os.path.exists(input_filename)):
                # logging.info(f'Reading {input_filename}')
                with bz2.open(input_filename) as input_file:
                        input_data = pd.read_pickle(input_file)
                        self.site_data[site_lc] = input_data 
                        # logging.info(site_lc)
                        # logging.info(self.site_data[site_lc])
                        # logging.info(f'\t{len(self.site_data[site_lc].columns)} columns, {len(self.site_data[site_lc])} rows')

                self.site_data_by_night[site_lc] = {}
                local_site_data = self.site_data[site_lc].iterrows()
                for _index, night in local_site_data:
                    night_start_line = night
                    night_date = night["Time_Stamp_UTC"].date()
                    night_date = night_date.strftime('%Y-%m-%d') 
                    night_list = [night_start_line]
                    preivous_line = night_start_line
                    _index2, current_line = next(local_site_data)

                    while((current_line["Time_Stamp_UTC"] - preivous_line["Time_Stamp_UTC"] < timedelta(hours= 7))):
                        night_list.append(current_line)
                        preivous_line = current_line
                        try:
                            _index3, current_line = next(local_site_data)
                        except StopIteration:
                            print("End of data")
                            break
                         
                    self.site_data_by_night[site_lc][night_date] = night_list
            else:   
                print(f'Error, {input_filename} not found')
        
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
    
    @staticmethod
    def data_file_path(filename: str) -> str:
        return os.path.join('..', '..', 'data', filename)

# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     env = Env()
#     print(env.site_data_by_night['gs']["2014-01-02"])