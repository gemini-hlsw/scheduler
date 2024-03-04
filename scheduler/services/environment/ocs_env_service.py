# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import bz2
import os
from typing import Dict, Final, FrozenSet, Optional

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import Angle
from astropy.time import Time
from lucupy.minimodel import ALL_SITES, Site, Variant, CloudCover, ImageQuality

from definitions import ROOT_DIR
from scheduler.services import logger_factory
from scheduler.services.abstract import ExternalService

logger = logger_factory.create_logger(__name__)


class OcsEnvService(ExternalService):
    """
    This is a historical Resource service used for OCS.
    It is based on data files provided by the science staff for the dates of 2018-01-01 to 2019-12-31.
    """

    # The columns used from the pandas dataframe.
    _time_stamp_col: Final[str] = 'Time_Stamp_UTC'
    _cc_band_col: Final[str] = 'raw_cc'
    _iq_band_col: Final[str] = 'raw_iq'
    _wind_speed_col: Final[str] = 'WindSpeed'
    _wind_dir_col: Final[str] = 'WindDir'

    # TODO: For time_slots that are not the default (1 min), we need to support passing the time_slot_length here.
    # TODO: It must be strictly in minutes for the pandas datasets to work since they are sampled for every minute.
    def __init__(self, sites: FrozenSet[Site] = ALL_SITES,
                 time_slot_length: Optional[int] = None):
        """
        Read in the pickled pandas data from the data files.
        Note that we assume that the data has been processed by the ocs-env-processor project at:
        https://github.com/gemini-hlsw/ocs-env-processor
        """
        self._sites = sites

        if time_slot_length is None:
            time_slot_length = 1
        self._time_slot_length = time_slot_length

        # The data per site. The pandas data structure is too complicated to fully represent.
        self._site_data: Dict[Site, pd.DataFrame] = {site: {} for site in self._sites}

        path = os.path.join(ROOT_DIR, 'scheduler', 'services', 'environment', 'data')
        for site in self._sites:
            site_lc = site.name.lower()
            input_filename = os.path.join(path, f'{site_lc}_weather_data.pickle.bz2')
            if not os.path.exists(input_filename) or not os.path.isfile(input_filename):
                raise FileNotFoundError(f'Could not find site weather data for {site.name}: '
                                        f'missing file {input_filename}')

            logger.info(f'Processing weather data for {site.name}...')
            with bz2.BZ2File(input_filename, 'rb') as input_file:
                df = pd.read_pickle(input_file)
                df[OcsEnvService._time_stamp_col] = pd.to_datetime(df[OcsEnvService._time_stamp_col])
                self._site_data[site] = df
                logger.info(f'Weather data for {site.name} read in: {len(self._site_data[site])} rows.')

    def get_actual_conditions_variant(self,
                                      site: Site,
                                      start_time: Time,
                                      end_time: Time) -> Optional[Variant]:
        """
        Return the weather variant.
        This should be site-based and time-based.
        times should be a contiguous set of times, but we do not force this.
        """
        df = self._site_data[site]

        # Now try using earliest and latest timestamp.
        # Convert AstroPy times to pandas UTC TimeStamps, taking the floor and ceil of the minutes since these
        # are converted from float jd values and thus are not quite accurate.
        # TODO: If these ever crash due to 1-off errors in length, try using .round('T') instead of floor / ceil, or
        # TODO: use seconds ('S') instead of minutes ('T'). Minutes are used since time slots are defined in minutes.
        start_time_timestamp = pd.to_datetime(start_time.datetime, utc=True).floor('T')
        end_time_timestamp = pd.to_datetime(end_time.datetime, utc=True).ceil('T')

        # Get all the entries between start and end.
        time_filtered_df = df[df[OcsEnvService._time_stamp_col].between(start_time_timestamp, end_time_timestamp)]

        # Take time slot length into account, taking only the 0 + n * time_slot_length entries for n ≥ 0.
        filtered_df = time_filtered_df.iloc[::self._time_slot_length]
        iq_array = np.array([ImageQuality(iq) for iq in filtered_df[OcsEnvService._iq_band_col].values],
                            dtype=ImageQuality)
        cc_array = np.array([CloudCover(cc) for cc in filtered_df[OcsEnvService._cc_band_col].values],
                            dtype=CloudCover)

        # TODO Performance: Remove units and just use numpy arrays?
        wind_dir_array = Angle(filtered_df[OcsEnvService._wind_dir_col].values, unit=u.deg)
        wind_spd_array = filtered_df[OcsEnvService._wind_speed_col].values * (u.m / u.s)

        return Variant(
            iq=iq_array,
            cc=cc_array,
            wind_dir=wind_dir_array,
            wind_spd=wind_spd_array
        )
