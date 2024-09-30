# Copyright (c) 2016-2024 Association of Universities for Research in Astronomy, Inc. (AURA)
# For license information see LICENSE or https://opensource.org/licenses/BSD-3-Clause

import bz2
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Final, FrozenSet, Optional

import astropy.units as u
import pandas as pd
from astropy.coordinates import Angle
from lucupy.minimodel import ALL_SITES, CloudCover, ImageQuality, Site, VariantSnapshot

from definitions import ROOT_DIR
from scheduler.services import logger_factory
from scheduler.services.abstract import ExternalService

logger = logger_factory.create_logger(__name__)


class OcsEnvService(ExternalService):
    """
    This is a historical Resource service used for OCS for Validation purposes.
    """

    # The columns used from the pandas dataframe.
    _night_time_stamp_col: Final[str] = 'Local_Night'
    _local_time_stamp_col: Final[str] = 'Local_Time'
    _cc_col: Final[str] = 'raw_cc'
    _iq_col: Final[str] = 'raw_iq'
    _wind_speed_col: Final[str] = 'WindSpeed'
    _wind_dir_col: Final[str] = 'WindDir'

    _night_time_col_initial: Final[str] = 'UT'
    _cc_col_initial: Final[str] = 'CC'
    _iq_col_initial: Final[str] = 'IQ'
    _wv_col_initial: Final[str] = 'WV'

    def __init__(self, sites: FrozenSet[Site] = ALL_SITES):
        """
        Read in the pickled pandas data from the data files.
        Note that we assume that the data has been processed by the ocs-env-processor project at:
        https://github.com/gemini-hlsw/ocs-env-processor
        """
        self._sites = sites

        # The data per site. The pandas data structure is too complicated to fully represent.
        self._site_data: Dict[Site, pd.DataFrame] = {}
        self._initial_conditions: Dict[Site, pd.DataFrame] = {}

        path = Path(ROOT_DIR) / 'scheduler' / 'services' / 'environment' / 'data' / 'validation'
        for site in self._sites:
            site_lc = site.name.lower()
            input_file_path = path / f'{site_lc}_weather_data.pickle.bz2'
            initial_conditions_path = path / f'{site_lc}_initial_conditions.csv'

            logger.debug(f'Processing weather data for {site.name}...')
            with bz2.BZ2File(input_file_path, 'rb') as input_file:
                df = pd.read_pickle(input_file)
                self._site_data[site] = df
                logger.debug(f'Weather data for {site.name} read in: {len(self._site_data[site])} rows.')

            with open(initial_conditions_path, 'r') as conditions_file:
                df = pd.read_csv(conditions_file)
                df[OcsEnvService._night_time_col_initial] = pd.to_datetime(df[OcsEnvService._night_time_col_initial],
                                                                           errors='coerce', format='%Y-%m-%d')
                self._initial_conditions[site] = df


    @staticmethod
    def _convert_to_variant(row) -> (datetime, VariantSnapshot):
        """
        Given a pandas row from the weather data, turn it into a Variant object.
        """
        timestamp = row[OcsEnvService._local_time_stamp_col].to_pydatetime()
        iq = ImageQuality(row[OcsEnvService._iq_col])
        cc = CloudCover(row[OcsEnvService._cc_col])
        wind_dir = Angle(row[OcsEnvService._wind_dir_col], unit=u.deg)
        wind_spd = row[OcsEnvService._wind_speed_col] * (u.m / u.s)

        variant_change = VariantSnapshot(iq=iq,
                                         cc=cc,
                                         wind_dir=wind_dir,
                                         wind_spd=wind_spd)
        return timestamp, variant_change

    def get_variant_changes_for_night(self,
                                      site: Site,
                                      night_date: date) -> Dict[datetime, VariantSnapshot]:
        """
        Return the weather variant.
        This should be site-based and time-based.
        times should be a contiguous set of times, but we do not force this.
        """
        df = self._site_data[site]

        # Get all the entries for the given night date.
        filtered_df = df[df[OcsEnvService._night_time_stamp_col].dt.date == night_date]
        variant_list = filtered_df.apply(OcsEnvService._convert_to_variant, axis=1).values.tolist()
        return {dt: v for dt, v in variant_list}

    def get_initial_conditions(self,
                               site: Site,
                               night_date: date) -> Optional[VariantSnapshot]:

        df = self._initial_conditions[site] if site in self._initial_conditions else None
        if not df.empty:

            filtered_df = df[df[OcsEnvService._night_time_col_initial].dt.date == night_date]

            df_iq = filtered_df[OcsEnvService._iq_col_initial].iloc[0]
            iq = ImageQuality.IQ70 if df_iq == 'na' else (ImageQuality(int(df_iq)/100) if df_iq != 'ANY' else ImageQuality.IQANY)
            df_cc = filtered_df[OcsEnvService._cc_col_initial].iloc[0]
            cc = CloudCover.CC70 if df_cc == 'na' else (CloudCover(int(df_cc)/100) if df_cc != 'ANY' else CloudCover.CCANY)

            return VariantSnapshot(iq=iq,
                                   cc=cc,
                                   wind_dir=Angle(292, unit=u.deg),
                                   wind_spd=3 * (u.m / u.s))
        return None
