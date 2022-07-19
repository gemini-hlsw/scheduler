
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

    _time_stamp = 'Time_Stamp_UTC'
    _day_difference = timedelta(hours=7)

    @staticmethod
    def _data_file_path(filename: str) -> str:
        """
        Create paths to files in the data directory.
        """
        return os.path.join('..', '..', 'data', filename)

    @staticmethod
    def _cc_band_to_float(data: str) -> float:
        """
        Returns max value from set of cc_band values 
        """
        if type(data) == str:
            new_value = data[1:-1].split(',')
            new_value = [float(s) for s in new_value]
            new_value = max(new_value) / 100
            return(new_value)
        elif pd.isna(data): 
            return np.nan
        else:
            return 1.0
    
    def __init__(self):
        """
        Stores weather data into a dictionary 
        that can be indexed by site and date.
        """

        PRODUCTION_MODE = False
            
        self.site_data_by_night = {}

        for site in Site:
            site_lc = site.name.lower()
            input_filename = Env._data_file_path(f'{site_lc}_wfs_filled_final_MEDIAN600s.pickle.bz2')
            output_filename = Env._data_file_path(f'{site_lc}_weather_data.pickle.bz2')

            if PRODUCTION_MODE and os.path.exists(output_filename):
                with bz2.open(output_filename) as output_file:
                    self.site_data_by_night[site] = pd.read_pickle(output_file)
                    logging.info(f'{site.site_name} data read in.')
                    continue

            logging.info(f'Processed data for site {site.site_name} not found.')
            logging.info(f'Attempting to process data from input file {input_filename}.')
            logging.info(f'Reading {input_filename}...')
            logging.info(f'Processing {site.name}')
          
            with bz2.open(input_filename) as input_file:
                input_data = pd.read_pickle(input_file)
                logging.info(f'\t\t{len(input_data.columns)} columns, {len(input_data)} rows')

                input_data = input_data.drop(columns={
                    'Night', f'{site.name}_Sun_Elevation', 'cc_requested', 'iq_requested',
                    'Airmass_QAP', 'Filter', 'Instrument', 'Object', 'Sidereal', 'Filename_QAP', 'Types',
                    'adaptive_optics', 'calibration_program', 'cass_rotator_pa', 'data_label', 'dec',
                    'engineering', 'exposure_time', 'filename', 'local_time', 'object', 'observation_class',
                    'observation_id', 'observation_type', 'program_id', 'ra', 'science_verification',
                    'types', 'wavefront_sensor', 'RA', 'DEC', 'Azimuth', 'Elevation', 'Airmass', 'Filename_FITS',
                    'IQ_OIWFS', 'IQ_P2WFS', 'IQ_OIWFS_MEDIAN', 'IQ_P2WFS_MEDIAN', 'QAP_IQ', 'QAP_IQ_WFS_scaled',
                    'Ratio_OIWFS', 'Ratio_P2WFS', 'QAP_IQ_WFS_scaled500', 'IQ_P2WFS_Zenith', 'IQ_OIWFS_Zenith',
                    'IQ_P2WFS_MEDIAN_Zenith', 'QAP_IQ_WFS_scaled500_Zenith'})

                self.site_data_by_night[site] = {}
                local_site_data = input_data.iterrows()
                for index, night in local_site_data:
                    night_start_line = night
                    night_date = night[Env._time_stamp].date()
                    night_start_line["cc_band"] = Env._cc_band_to_float(night_start_line["cc_band"])
                    night_start_line["iq_band"] /= 100
                    logging.info(f'\tProccesing UTC night of {night_date}')
                    night_list = [night_start_line]
                    previous_line = night_start_line
                    index2, current_line = next(local_site_data)
                    current_line["cc_band"] = Env._cc_band_to_float(current_line["cc_band"])
                    current_line["iq_band"] /= 100

                    while current_line[Env._time_stamp] - previous_line[Env._time_stamp] < Env._day_difference:
                        night_list.append(current_line)
                        previous_line = current_line
                        try:
                            index3, current_line = next(local_site_data)
                            current_line["cc_band"] = Env._cc_band_to_float(current_line["cc_band"])
                            current_line["iq_band"] /= 100
                        except StopIteration:
                            logging.info("End of data")
                            break
                
                    self.site_data_by_night[site][night_date] = night_list
            
                for night in self.site_data_by_night[site]:
                    self.site_data_by_night[site][night] = pd.DataFrame(self.site_data_by_night[site][night])
                    iq_band = self.site_data_by_night[site][night].iloc[0]["iq_band"]
                    cc_band = self.site_data_by_night[site][night].iloc[0]["cc_band"]
                    starting_index = self.site_data_by_night[site][night].index[0]

                    if pd.isna(iq_band):
                        self.site_data_by_night[site][night].at[starting_index, "iq_band"] = 1.0

                    if pd.isna(cc_band):
                        self.site_data_by_night[site][night].at[starting_index, "cc_band"] = 1.0

                    self.site_data_by_night[site][night] = self.site_data_by_night[site][night].fillna(method="ffill")
                
                logging.info(f'Writing {output_filename}')
                pd.to_pickle(self.site_data_by_night[site], output_filename)
        
    @staticmethod
    def get_actual_conditions_variant(site: Site,
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
    # print(env.site_data_by_night[Site.GN][date(2020, 1, 1)])