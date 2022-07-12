
import logging
import bz2
import os
from datetime import date, datetime, timedelta
from typing import Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle
from astropy.time import Time
from astropy.units import Quantity

from common.minimodel import Site, Variant, CloudCover, ImageQuality, WaterVapor


class Env:
    def __init__(self):
        return
        """
        Stores weather data into a dictionary 
        that can be indexed by site and date.
        """

        time_stamp = 'Time_Stamp_UTC'
        day_difference = timedelta(hours=7)

        def data_file_path(filename: str) -> str:
            return os.path.join('..', '..', 'data', filename)

        self.site_data_by_night = {}

        for site in Site:
            logging.info(f'Processing {site.name}')
            input_filename = data_file_path(f'{site.name.lower()}_weather_data.pickle.bz2')

            logging.info(f'Reading {input_filename}')
            with bz2.open(input_filename) as input_file:
                input_data = pd.read_pickle(input_file)
                logging.info(f'\t\t{len(input_data.columns)} columns, {len(input_data)} rows')

            self.site_data_by_night[site] = {}
            local_site_data = input_data.iterrows()
            for index, night in local_site_data:
                night_start_line = night

                night_date = night[time_stamp].date()
                logging.info(f'\tProccesing UTC night of {night_date}')

                night_list = [night_start_line]
                previous_line = night_start_line
                index2, current_line = next(local_site_data)

                while current_line[time_stamp] - previous_line[time_stamp] < day_difference:
                    night_list.append(current_line)
                    previous_line = current_line
                    try:
                        index3, current_line = next(local_site_data)
                    except StopIteration:
                        logging.info("End of data")
                        break
                        
                self.site_data_by_night[site][night_date] = night_list
            
        
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
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env = Env()

    # print(env.site_data_by_night[Site.GS][date(2014, 1, 2)])