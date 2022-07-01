import bz2
import os
import pandas as pd

if __name__ == '__main__':
    filename = os.path.join('..', 'data', 'gn_weather_data.pickle.bz2')
    with bz2.open(filename) as input_file:
        input_data = pd.read_pickle(input_file)
        # for i in input_data:
        #     print(i)
        # input_data = input_data.fillna(method="pad")
        # print(input_data.loc[[0]])
        # print(input_data.loc[[1]])
        # print(input_data.loc[[2]])
        print(input_data[['Time_Stamp_UTC', 'cc_band', 'iq_band', 'WindDir', 'WindSpeed']].to_string())
        # print(input_data['cc_band'].notna())