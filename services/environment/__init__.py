import bz2
from copy import copy
import logging
from math import ceil
import os
from datetime import timedelta, date, datetime
from typing import Dict, List, Union

from astropy import units as u
import numpy as np
import pandas as pd
import strawberry
import uvicorn
from fastapi import FastAPI
from strawberry.asgi import GraphQL

from astropy.coordinates import Angle

from lucupy.minimodel import Site, Variant, CloudCover, ImageQuality


class Env:
    _time_stamp = 'Time_Stamp_UTC'
    _day_difference = timedelta(hours=7)
    _PRODUCTION_MODE = True
    _cc_band = 'cc_band'
    _iq_band = 'iq_band'
    _wind_speed = 'WindSpeed'
    _wind_dir = 'WindDir'

    @staticmethod
    def _data_file_path(filename: str) -> str:
        """
        Create paths to files in the data directory.
        """
        return os.path.join('data', filename)

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
        Loads weather data into a dictionary if processed data currently exists.
        If data does not exist or reprocessing is requested, data if reprocessed for current
        and future use (note that this is a slow process).

        Weather data is stored into a dictionary that can be indexed by site and date.
        """
        self.site_data_by_night: Dict[Site, Dict[date]] = {}

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

                # The time slot length as a Timedelta.
                td = pd.Timedelta(1, 'm')

                # These columns do not appear to be used here, but if we remove them, we
                # end up with rows with only NaN data, which seems to mess up the timestamp.
                # input_data = input_data.drop(columns={
                #     'cc_extinction', 'cc_extinction_error', 'iq_delivered', 'iq_delivered_error',
                #     'iq_zenith', 'ut_time', 'Time_Stamp', 'Wavelength', 'azimuth',
                #     'elevation', 'filter_name', 'instrument', 'telescope',
                #     'Relative_Humidity', 'Temperature', 'Dewpoint', 'iq_delivered500',
                #     'IQ_OIWFS_MEDIAN_Zenith', 'iq_delivered500_Zenith',
                #     'iq_zenith_error', 'airmass'
                # })
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

                        # Process the relevant data so that they are defined for the first entry
                        # of the night.
                        if pd.isna(cur_row[Env._iq_band]):
                            cur_row[Env._iq_band] = 1.0
                        else:
                            cur_row[Env._iq_band] /= 100

                        if pd.isna(cur_row[Env._cc_band]):
                            cur_row[Env._cc_band] = 1.0
                        else:
                            cur_row[Env._cc_band] = Env._cc_band_to_float(cur_row[Env._cc_band])

                        if pd.isna(cur_row[Env._wind_speed]):
                            cur_row[Env._wind_speed] = 1.0
                        if pd.isna(cur_row[Env._wind_dir]):
                            cur_row[Env._wind_dir] = 1.0

                        # Set prev_row to None to indicate no previous data for this day.
                        prev_row = None
                    else:
                        # Process the iq_band if it exists by dividing it by 100 to bin it properly.
                        if not pd.isna(cur_row[Env._iq_band]):
                            cur_row[Env._iq_band] = float(cur_row[Env._iq_band]) / 100

                        # Process the cc_band as it could be a set, in which case, we want the maximum value.
                        cur_row[Env._cc_band] = Env._cc_band_to_float(cur_row[Env._cc_band])

                    # Calculate the time difference between this row and the previous row, if there was a previous row.
                    if prev_row is None:
                        timediff = 0
                    else:
                        cur_time = cur_row[Env._time_stamp]
                        prev_time = prev_row[Env._time_stamp]
                        timediff = (cur_time - prev_time).seconds / 60
                        if timediff != ceil(timediff):
                            raise ValueError('timediff is not a value in minutes: '
                                             f'{cur_time} - {prev_time} = {timediff} minutes')
                        timediff = int(timediff)

                    # Add empty rows to account for missed measurements.
                    for idx in range(timediff - 1):
                        # Copy the previous row since we want to make a new row with a new time stamp.
                        prev_row = copy(prev_row)
                        prev_row[Env._time_stamp] += td
                        night_rows.append(prev_row)

                    night_rows.append(cur_row)
                    prev_row = cur_row

                # Add the last day, which has not been added yet due to the loop ending.
                self.site_data_by_night[site][night_date] = night_rows

                # Now we have all the data broken into nights on a minute by minute basis.
                logging.info('Filling in missing CC, IQ, WindSpeed, and WindDir information...')
                for night in self.site_data_by_night[site]:
                    # Convert to data frame.
                    self.site_data_by_night[site][night] = pd.DataFrame(self.site_data_by_night[site][night])

                    # Fill in missing data from the previous populated entry for iq_band and cc_band.
                    self.site_data_by_night[site][night][[Env._cc_band, Env._iq_band,
                                                          Env._wind_speed, Env._wind_dir]] = (
                        self.site_data_by_night[site][night][[Env._cc_band, Env._iq_band,
                                                              Env._wind_speed, Env._wind_dir]].ffill()
                    )

                logging.info('Processing done.')
                logging.info(f'Writing {output_filename}')
                pd.to_pickle(self.site_data_by_night[site], output_filename)

    def get_weather(self, site: Site, start_time: datetime, end_time: datetime, time_slot_length: int) -> List[Variant]:
        """
        Returns list of weather data
        based off start and end times 
        """
        if time_slot_length < 1:
            raise ValueError(f'Time slot length must be a positive integer: {time_slot_length}')
        if start_time > end_time:
            raise ValueError(f'Invalid time range: {start_time} to {end_time}.')

        weather_list = []
        variant_list = []

        start_date = start_time.date()
        end_date = end_time.date()
        nights = [n for n in self.site_data_by_night[site] if start_date <= n <= end_date]

        for night in nights:
            night_list = []
            for _, data in self.site_data_by_night[site][night].iterrows():
                if start_time <= data[Env._time_stamp] <= end_time:
                    night_list.append(data)

            # We only want all entries from each night of the form:
            # first entry + x * time_slot_length for all x >= 0.
            weather_list.extend(night_list[::time_slot_length])
        pd.DataFrame(weather_list)

        for weather in weather_list:
            # extract the fields above
            try:
                variant = Variant(
                    start_time=weather[Env._time_stamp],
                    cc=CloudCover(weather[Env._cc_band]),
                    iq=ImageQuality(weather[Env._iq_band]),
                    wind_dir=Angle(weather[Env._wind_dir] * u.rad),
                    wind_spd=weather[Env._wind_speed] * u.m / u.s
                )
                variant_list.append(variant)

            except ValueError as e:
                logging.error(f'get_weather: {e}')

        return variant_list


@strawberry.type
class SVariant:
    """
    Variant data for query service 
    """

    start_time: datetime
    iq: ImageQuality
    cc: CloudCover
    wind_dir: float
    wind_spd: float

    @staticmethod
    def from_computed_variant(variant: Variant) -> 'SVariant':
        return SVariant(
            start_time=variant.start_time,
            iq=variant.iq,
            cc=variant.cc,
            wind_dir=variant.wind_dir.value,
            wind_spd=variant.wind_spd.value
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    env = Env()

    # To access server, go to: http://127.0.0.1:8000/graphql
    # Here is an example query:
    # {
    #     weather(site: GN, startDate: "2016-07-20T12:00:00", endDate: "2016-09-01T12:00:00", timeSlotLength: 5) {
    #         startTime
    #         iq
    #         cc
    #         windDir
    #         windSpd
    #     }
    # }

    @strawberry.type
    class Query:
        @strawberry.field
        def weather(self, site: Site,
                    start_date: datetime,
                    end_date: datetime,
                    time_slot_length: int = 1) -> List[SVariant]:
            svariant_list = []
            variant_list = env.get_weather(site, start_date, end_date, time_slot_length)
            for variant in variant_list:
                svariant_list.append(SVariant.from_computed_variant(variant))

            return svariant_list

    schema = strawberry.Schema(query=Query)
    graphql_app = GraphQL(schema)
    app = FastAPI()
    app.add_route('/graphql', graphql_app)
    app.add_websocket_route('/graphql', graphql_app)
    uvicorn.run(app, host='127.0.0.1', port=8000)
