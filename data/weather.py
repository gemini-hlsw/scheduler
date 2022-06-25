# Need to run "pip install pandas" first in terminal
import bz2
import pandas as pd
import os

if __name__ == '__main__':
    gn_input_filename = os.path.join('..', 'data', 'gn_wfs_filled_final_MEDIAN600s.pickle.bz2')
    gs_input_filename = os.path.join('..', 'data', 'gs_wfs_filled_final_MEDIAN600s.pickle.bz2')  
    gn_output_file = os.path.join('..', 'data', 'gn_weather_data.pickle.bz2')
    gs_output_file = os.path.join('..', 'data', 'gs_weather_data.pickle.bz2')

    with bz2.open(gn_input_filename) as gnf:
        data = pd.read_pickle(gnf)
        data = data.drop(columns=['Night', 'GN_Sun_Elevation', 'cc_requested', 'iq_requested',
        'Airmass_QAP','Filter', 'Instrument', 'Object', 'Sidereal', 'Filename_QAP', 'Types', 
        'adaptive_optics', 'calibration_program', 'cass_rotator_pa', 'data_label', 'dec',
        'engineering', 'exposure_time', 'filename', 'local_time', 'object', 'observation_class',
        'observation_id', 'observation_type', 'program_id', 'ra', 'science_verification',
        'types', 'wavefront_sensor', 'RA', 'DEC', 'Azimuth', 'Elevation', 'Airmass', 'Filename_FITS',
        'IQ_OIWFS', 'IQ_P2WFS', 'IQ_OIWFS_MEDIAN', 'IQ_P2WFS_MEDIAN', 'QAP_IQ', 'QAP_IQ_WFS_scaled',
        'Ratio_OIWFS', 'Ratio_P2WFS', 'QAP_IQ_WFS_scaled500', 'IQ_P2WFS_Zenith', 'IQ_OIWFS_Zenith',
        'IQ_P2WFS_MEDIAN_Zenith', 'QAP_IQ_WFS_scaled500_Zenith'])
        pd.to_pickle(data, gn_output_file)
        print('***** GEMINI NORTH DATA *****')
        print(data)
        print('\n\n')
       
    with bz2.open(gs_input_filename) as gsf:
        data = pd.read_pickle(gsf)
        data = data.drop(columns=['Night', 'GS_Sun_Elevation', 'cc_requested', 'iq_requested',
        'Airmass_QAP','Filter', 'Instrument', 'Object', 'Sidereal', 'Filename_QAP', 'Types', 
        'adaptive_optics', 'calibration_program', 'cass_rotator_pa', 'data_label', 'dec',
        'engineering', 'exposure_time', 'filename', 'local_time', 'object', 'observation_class',
        'observation_id', 'observation_type', 'program_id', 'ra', 'science_verification',
        'types', 'wavefront_sensor', 'RA', 'DEC', 'Azimuth', 'Elevation', 'Airmass', 'Filename_FITS',
        'IQ_OIWFS', 'IQ_P2WFS', 'IQ_OIWFS_MEDIAN', 'IQ_P2WFS_MEDIAN', 'QAP_IQ', 'QAP_IQ_WFS_scaled',
        'Ratio_OIWFS', 'Ratio_P2WFS', 'QAP_IQ_WFS_scaled500', 'IQ_P2WFS_Zenith', 'IQ_OIWFS_Zenith',
        'IQ_P2WFS_MEDIAN_Zenith', 'QAP_IQ_WFS_scaled500_Zenith'])
        pd.to_pickle(data, gs_output_file)
        print('***** GEMINI SOUTH DATA *****')
        print(data)
        

 