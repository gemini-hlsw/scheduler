import bz2
import os

import pandas as pd

if __name__ == '__main__':
    def print_processed():
        filename = os.path.join('..', 'data', 'gn_weather_data.pickle.bz2')
        with bz2.open(filename) as input_file:
            input_data = pd.read_pickle(input_file)

        for key in list(input_data.keys())[:4]:
            # input_data[key] = input_data[key].drop(columns={
            #     'cc_extinction', 'cc_extinction_error', 'iq_delivered', 'iq_delivered_error',
            #     'iq_zenith', 'ut_time', 'Time_Stamp', 'Wavelength', 'azimuth',
            #     'elevation', 'filter_name', 'instrument', 'telescope',
            #     'Relative_Humidity', 'Temperature', 'Dewpoint', 'iq_delivered500',
            #     'IQ_OIWFS_MEDIAN_Zenith', 'iq_delivered500_Zenith',
            #     'iq_zenith_error', 'airmass'
            # })
            print(input_data[key].loc[:].to_string())

        # print(input_data.loc[[1]])
        # print(input_data.loc[[2]])
        # print(input_data[['Time_Stamp_UTC', 'cc_band', 'iq_band', 'WindDir', 'WindSpeed']].to_string())
        # print(input_data['cc_band'].notna())


    def print_unprocessed():
        filename = os.path.join('..', 'data', 'gn_wfs_filled_final_MEDIAN600s.pickle.bz2')
        with bz2.open(filename) as input_file:
            input_data = pd.read_pickle(input_file)
            input_data = input_data.drop(columns={
                'Night', f'GN_Sun_Elevation', 'cc_requested', 'iq_requested',
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
            print(input_data.loc[0:1145].to_string())


    print_processed()
