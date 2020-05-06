import json
import pandas
import sys
import os
import numpy as np
import re
import helpers


# args
# input_feature_file = "../data/indices/UK100.csv"
# output_file = "../results/all-features-UK100.csv"
# symbol = "UK100"

input_feature_file = "../data/indices/FTSE100.csv"
output_file = "../results/all-features-FTSE100.csv"
symbol = "FTSE100"

input_directory = "../results/data"
input_file_metrics = ["Price","Open","High","Low","Vol"]

# models = ["arma","arima","lstm","cnn","lstmcnn_wsum_combination","lstmcnn_wsum_layer_lstmDlayer","lstmcnn_wsum_layer"]
models = ["arma","lstm","cnn","lstmcnn_wsum_combination","lstmcnn_wsum_layer_lstmDlayer","lstmcnn_wsum_layer"]



input_dataframe = pandas.read_csv(input_feature_file)
timestamp = np.array(input_dataframe['timestamp'])
data = {'Close':np.array(input_dataframe['Close']),
        'Open':np.array(input_dataframe['Open']),
        'High':np.array(input_dataframe['High']),
        'Low':np.array(input_dataframe['Low']),
        'Vol':np.array(input_dataframe['Vol']),
        # 'label':np.array(input_dataframe['label']),
        # 'label_title':np.array(input_dataframe['label_title']),
        # 'label_url':np.array(input_dataframe['label_url']),
        # 'label_source':np.array(input_dataframe['label_source']),
        }
feature_order = ['Close',
                'Open',
                'High',
                'Low',
                'Vol'
                # 'label',
                # 'label_title',
                # 'label_url',
                # 'label_source'
                ]

index_input_directory = "../data/indices"
reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(index_input_directory):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])
csv_input_files.sort()

# for input_file in csv_input_files:
#     if not symbol in input_file:
#         continue
#     input_dataframe = pandas.read_csv(input_file)
#     features = ["Close","Open","High","Low","Vol"]
#     print(len(np.array(input_dataframe["Close"])))
#     for f in features:
#         feature_label = input_file.split('.csv')[0] + '.' + f
#         data[feature_label] = np.array(input_dataframe[f])
#         feature_order.append(feature_label)
#         print(feature_label)

for m in models:
    model_input_directory = input_directory + "/" + m
    # get all csv files in input directory
    reg_x = re.compile(r'\.(csv)')
    csv_input_files = []
    for path, dnames, fnames in os.walk(model_input_directory):
        csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])
    csv_input_files.sort()

    for input_file in csv_input_files:
        input_dataframe = pandas.read_csv(input_file)

        if not symbol in input_file:
            continue
        # if not "Volume" in input_file:
        #     continue
        if "lse_2009-07-24_2020-03-20" in input_file:
            prefix = "lse"
        else:
            prefix = input_file.split("/")[-1].split(".metric-")[0]
        

        feature_label = m + "." + prefix +".warp_distance." + input_file.split("/")[-1].split(".metric-")[1].split(".csv")[0] 
        try:
            input_dataframe['warp_distance']
        except KeyError:
            print('!! NOT FOUND : '+feature_label )
            continue 
        data[feature_label] = np.array(input_dataframe['warp_distance'])
        feature_order.append(feature_label)
        print(feature_label)


out_dataframe = pandas.DataFrame(data, index=timestamp)
out_dataframe.index.name = "timestamp"
out_dataframe = out_dataframe[feature_order]
out_dataframe.to_csv(output_file)



