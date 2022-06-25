import bz2
import logging
import os

import pandas as pd

from common.minimodel import Site

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    def data_file_path(filename: str) -> str:
        return os.path.join('..', 'data', filename)

    for site in Site:
        site_lc = site.name.lower()
        input_filename = data_file_path(f'{site_lc}_wfs_filled_final_MEDIAN600s.pickle.bz2')
        output_filename = data_file_path(f'{site_lc}_weather_data.pickle.bz2')

        logging.info(f'Reading {input_filename}')
        with bz2.open(input_filename) as input_file:
            input_data = pd.read_pickle(input_file)
            logging.info(f'\t{len(input_data.columns)} columns, {len(input_data)} rows')
   
            # GN has GN_Sun_Elevation, GS has GS_Sun_Elevation.
            output_data = input_data.drop(columns={ 
                'Night', f'{site.name}_Sun_Elevation', 'cc_requested', 'iq_requested',
                'Airmass_QAP', 'Filter', 'Instrument', 'Object', 'Sidereal', 'Filename_QAP', 'Types',
                'adaptive_optics', 'calibration_program', 'cass_rotator_pa', 'data_label', 'dec',
                'engineering', 'exposure_time', 'filename', 'local_time', 'object', 'observation_class',
                'observation_id', 'observation_type', 'program_id', 'ra', 'science_verification',
                'types', 'wavefront_sensor', 'RA', 'DEC', 'Azimuth', 'Elevation', 'Airmass', 'Filename_FITS',
                'IQ_OIWFS', 'IQ_P2WFS', 'IQ_OIWFS_MEDIAN', 'IQ_P2WFS_MEDIAN', 'QAP_IQ', 'QAP_IQ_WFS_scaled',
                'Ratio_OIWFS', 'Ratio_P2WFS', 'QAP_IQ_WFS_scaled500', 'IQ_P2WFS_Zenith', 'IQ_OIWFS_Zenith',
                'IQ_P2WFS_MEDIAN_Zenith', 'QAP_IQ_WFS_scaled500_Zenith'})

            logging.info(f'Writing {output_filename}')
            logging.info(f'\t{len(output_data.columns)} columns, {len(input_data)} rows')
            pd.to_pickle(output_data, output_filename)
    