import json
import pandas
import math
import sys
import os
import numpy as np
import re



# args
input_directory = "../data/lseg"
label_window_size = 3
training_ends = 341 #2012-12-13

# get all csv files in input directory
reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(input_directory):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])
csv_input_files.sort()



for input_file in csv_input_files:
    print("Processing " + input_file)
    
    input_dataframe = pandas.read_csv(input_file)

    # Close = np.array(input_dataframe['Close'])
    # Open = np.array(input_dataframe['Open'])
    # High = np.array(input_dataframe['High'])
    # Low = np.array(input_dataframe['Low'])
    Vol = np.array(input_dataframe['Vol'])
    label = np.array(input_dataframe['label'])
    timestamp = np.array(input_dataframe['timestamp'])

  

    #Cleaning traing window
    for i in range(0,training_ends):
        if label[i] == 1:
            Vol = np.delete(Vol, i)
            label = np.delete(label, i)
            timestamp = np.delete(timestamp, i)
    
    #Expanding label
    for i in range(0, len(label)):
        if label[i] == 1:
            for j in range(1,label_window_size):
                if i-j >= 0:
                    label[i-j] = 1
            

    data = {
            # 'Close':Close,
            # 'Open':Open,
            # 'High':High,
            # 'Low':Low,
            'Vol':Vol,
            'label':label
            }
    out_dataframe = pandas.DataFrame(data, index=timestamp)
    out_dataframe.index.name = "timestamp"
    out_dataframe = out_dataframe[[
                            # 'Close',
                            # 'Open',
                            # 'High',
                            # 'Low',
                            'Vol',
                            'label'
                            ]]
    
    out_dataframe.to_csv(input_file)




