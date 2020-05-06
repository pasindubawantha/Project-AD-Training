import json
import pandas
import math
import sys
import os
import numpy as np


# args
input_file = "../data/lseg/lse_High_uk100_labeled.csv"
label_file = "../data/lse-uk100-warpdistance-differance-outliers.csv"

label_dataframe = pandas.read_csv(label_file)
label = np.array(label_dataframe['label'])



print("Processing " + input_file)

input_dataframe = pandas.read_csv(input_file)

# Close = np.array(input_dataframe['Close'])
# Open = np.array(input_dataframe['Open'])
High = np.array(input_dataframe['High'])
# Low = np.array(input_dataframe['Low'])
# Vol = np.array(input_dataframe['Vol'])
timestamp = np.array(input_dataframe['timestamp'])

data = {
        # 'Close':Close,
        # 'Open':Open,
        'High':High,
        # 'Low':Low,
        # 'Vol':Vol,
        'label':label
        }
out_dataframe = pandas.DataFrame(data, index=timestamp)
out_dataframe.index.name = "timestamp"
out_dataframe = out_dataframe[[
                        # 'Close',
                        # 'Open',
                        'High',
                        # 'Low',
                        # 'Vol',
                        'label'
                        ]]

out_dataframe.to_csv(input_file)




