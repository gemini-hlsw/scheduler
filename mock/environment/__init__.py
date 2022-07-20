import bz2
import logging
import os
from datetime import timedelta
from typing import Optional, Union

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
    _PRODUCTION_MODE = False
    _cc_band = 'cc_band'
    _iq_band = 'iq_band'

    @staticmethod
    def _data_file_path(filename: str) -> str:
        """
        Create paths to files in the data directory.
        """
        return os.path.join('..', '..', 'data', filename)

    @staticmethod
    def _cc_band_to_float(data: Union[str, float]) -> float:
        """
        Returns max value from set of cc_band values 
        """
        # If it is a float, handle it appropriately.
        if pd.isna(data):
            return np.nan

        # Otherwise, it is a str. If it is a set, eval it to get the set, and return the max.
        if type(data) == str and '{' in data:
            return max(eval(data)) / 100

        return float(data) / 100

    def __init__(self):
        """
        Stores weather data into a dictionary 
        that can be indexed by site and date.
        """
        self.site_data_by_night = {}

        for site in Site:
            site_lc = site.name.lower()
            input_filename = Env._data_file_path(f'{site_lc}_wfs_filled_final_MEDIAN600s.pickle.bz2')
            output_filename = Env._data_file_path(f'{site_lc}_weather_data.pickle.bz2')

            logging.info(f'Processing {site.name}...')
            if Env._PRODUCTION_MODE and os.path.exists(output_filename):
                with bz2.open(output_filename) as output_file:
                    self.site_data_by_night[site] = pd.read_pickle(output_file)
                    logging.info(f'{site.site_name} data read in.')
                    continue

            logging.info(f'Processed data for site {site.site_name} not found.')
            logging.info(f'Attempting to process data from input file {input_filename}.')

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
                input_data = input_data.drop(columns={
                    'cc_extinction', 'cc_extinction_error', 'iq_delivered', 'iq_delivered_error',
                    'iq_zenith', 'ut_time', 'Time_Stamp', 'Wavelength', 'azimuth',
                    'elevation', 'filter_name', 'instrument', 'telescope',
                    'Relative_Humidity', 'Temperature', 'Dewpoint', 'iq_delivered500',
                    'IQ_OIWFS_MEDIAN_Zenith', 'iq_delivered500_Zenith',
                    'iq_zenith_error', 'airmass'
                })
                self.site_data_by_night[site] = {}

                # We first divide the data into a separate dataframe per night.
                # 1. night_date of None to indicate no nights have been processed.
                # 2. An empty list of night_rows that will hold the data rows for the night we are working on.
                # 3. prev_row of None to indicate that there has been no previously processed night row, since
                #    a new night will begin when _day_difference has passed from the previous night row.
                night_date = None
                night_rows = []
                prev_row = None

                for index, cur_row in input_data.iterrows():
                    # Check if we are starting a new night.
                    if (night_date is None or
                            (prev_row is not None and
                             cur_row[Env._time_stamp] - prev_row[Env._time_stamp] >= Env._day_difference)):
                        # If we have former night data, add it to the processed data.
                        if night_date is not None:
                            self.site_data_by_night[site][night_date] = night_rows

                        # Now proceed to start the next night.
                        night_date = cur_row[Env._time_stamp].date()
                        logging.info(f'\tProcessing UTC night of {night_date}')
                        night_rows = []

                        # Process the iq_band and cc_band so that they are defined for the first entry
                        # of the night.
                        if pd.isna(cur_row[Env._iq_band]):
                            cur_row[Env._iq_band] = 1.0
                        else:
                            cur_row[Env._iq_band] /= 100

                        # Process the cc_band.
                        if pd.isna(cur_row[Env._cc_band]):
                            cur_row[Env._cc_band] = 1.0
                        else:
                            cur_row[Env._cc_band] = Env._cc_band_to_float(cur_row[Env._cc_band])
                    else:
                        # Process the iq_band if it exists by dividing it by 100 to bin it properly.
                        if not pd.isna(cur_row[Env._iq_band]):
                            cur_row[Env._iq_band] = float(cur_row[Env._iq_band]) / 100

                        # Process the cc_band as it could be a set, in which case, we want the maximum value.
                        cur_row[Env._cc_band] = Env._cc_band_to_float(cur_row[Env._cc_band])

                    # Add the new row to the night.
                    night_rows.append(cur_row)
                    prev_row = cur_row

                # Add the last day, which has not been added yet due to the loop ending.
                self.site_data_by_night[site][night_date] = night_rows

                # Now we have all the data broken into nights on a minute by minute basis.
                logging.info('Filling in missing CC and IQ information...')
                for night in self.site_data_by_night[site]:
                    # Convert to data frame.
                    self.site_data_by_night[site][night] = pd.DataFrame(self.site_data_by_night[site][night])

                    # Fill in missing data from the previous populated entry for iq_band and cc_band..
                    self.site_data_by_night[site][night] = self.site_data_by_night[site][night].fillna(method="ffill")

                logging.info('Processing done.')
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
