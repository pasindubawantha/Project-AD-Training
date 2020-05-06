import json
import pandas
import math
import sys
import os
import numpy as np
import re
import helpers
import csv


# args
input_directory = "../data/indices"
label_window_half = 2
data_time_input_format = '%b %d, %Y'
data_time_output_format = '%Y-%m-%d'


# get all csv files in input directory
reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(input_directory):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])
csv_input_files.sort()



for input_file in csv_input_files:
    print("Processing " + input_file)
    
    input_dataframe = pandas.read_csv(input_file)

    timestamp = np.array(input_dataframe['Date'])
    Close = np.array(input_dataframe['Price'])
    Open = np.array(input_dataframe['Open'])
    High = np.array(input_dataframe['High'])
    Low = np.array(input_dataframe['Low'])
    Vol = np.array(input_dataframe['Vol.'])
    # label = np.array(input_dataframe['label'])
    # label_title = np.array(input_dataframe['label_title'])
    # label_url = np.array(input_dataframe['label_url'])
    # label_source = np.array(input_dataframe['label_source'])

    # fix volume
    for i in range(0,len(Vol)):
        if Vol[i] == '-':
            Vol[i] = 0
            # print(Vol[i])
        elif Vol[i][-1] == 'K':
            Vol[i] = int(float(Vol[i].replace(",", "").replace("K", ""))*1000)
            # print(Vol[i])
        elif Vol[i][-1] == 'M':
            Vol[i] = int(float(Vol[i].replace(",", "").replace("M", ""))*1000000)
            # print(Vol[i])
        elif Vol[i][-1] == 'B':
            Vol[i] = int(float(Vol[i].replace(",", "").replace("B", ""))*1000000000)
            # print(Vol[i])
        else:
            print(" !! Unformated Volume !! ")
            print(Vol[i])

    # fix timestamp
    timestamp = helpers.string_to_date(timestamp, data_time_input_format)
    timestamp = helpers.date_to_string(timestamp, data_time_output_format)
    # print(timestamp)

    data = {'Close':Close,
            'Open':Open,
            'High':High,
            'Low':Low,
            'Vol':Vol
            # 'label':label,
            # 'label_title':label_title,
            # 'label_url':label_url,
            # 'label_source':label_source
            }
    out_dataframe = pandas.DataFrame(data, index=timestamp)
    out_dataframe.index.name = "timestamp"
    out_dataframe = out_dataframe[['Close',
                            'Open',
                            'High',
                            'Low',
                            'Vol',
                            # 'label',
                            # 'label_title',
                            # 'label_url',
                            # 'label_source'
                            ]]
    

    
    out_dataframe.to_csv(input_file)
print("All Done !")



