import json
import pandas
import math
import sys
import os
import numpy as np
import re
import helpers
import detector.detector as detector
import detector.confusion_metrics as confusion_metrics
from datetime import datetime
from datetime import timedelta 


# args
input_directory = "../results/data"
input_file_metrics = ["Price","Open","High","Low"]
input_summary_file = "../data/lse_summary.csv"
output_directory = "../results"
new_file = "../data/lse-news.csv"
process_file_limit = -1
label_window_half = 2

models = ["arma","arima","lstm","cnn","lstmcnn_wsum_combination","lstmcnn_wsum_layer_lstmDlayer","lstmcnn_wsum_layer"]
# models = ["arma"]


input_summary = pandas.read_csv(input_summary_file, index_col="file")

news_dataframe = pandas.read_csv(new_file)
news_timestamp = np.array(news_dataframe['Date'])

news_timestamp = helpers.string_to_date(news_timestamp, '%b %d, %Y')

data = {'title':np.array(news_dataframe['Title']),
        'source':np.array(news_dataframe['Source']),
        'url':np.array(news_dataframe['Url']),
        'market_closed':np.zeros(len(news_timestamp))}
news_dataframe = pandas.DataFrame(data, index=news_timestamp)
news_dataframe.index.name = "timestamp"
news_dataframe = news_dataframe[['title',
                        'source',
                        'url',
                        'market_closed']]        

for m in models:
    model_input_directory = input_directory + "/" + m
    # get all csv files in input directory
    reg_x = re.compile(r'\.(csv)')
    csv_input_files = []
    for path, dnames, fnames in os.walk(model_input_directory):
        csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])
    csv_input_files.sort()
    model_file_name = output_directory + "/" + m + "_list.csv"
    model_dataframe = pandas.read_csv(model_file_name, index_col="file")

    data = {'mse':model_dataframe['mse'], 
            'TP':model_dataframe['TP'],
            'FP':model_dataframe['FP'],
            'FN':model_dataframe['FN'],
            'TN':model_dataframe['TN'],
            'parameters':model_dataframe['parameters'],
            'threshold_parameters':model_dataframe['threshold_parameters'],
            'no_of_anomalies':model_dataframe['no_of_anomalies'],
            'first_label':model_dataframe['first_label'],
            'length':model_dataframe['length'],
            "first_label_ratio":model_dataframe['first_label_ratio']}
    model_dataframe = pandas.DataFrame(data)
    model_dataframe.index.name = "file"
    model_dataframe = model_dataframe[['mse',
                                        'TP',
                                        'FP',
                                        'FN',
                                        'TN',
                                        'parameters',
                                        'threshold_parameters',
                                        'no_of_anomalies',
                                        'first_label',
                                        'length',
                                        "first_label_ratio"]]
    
    print("##### ["+m+"] "+ str(len(csv_input_files)) + " CSV input files to process #####")
    count = 1
    for input_file in csv_input_files:
        print("Processing ["+m+"]" + input_file)
        file_name = input_file.split("/")[-1]
        file_name_in_summary = file_name.split(".metric-")[0] + ".csv"
        print("File name : " + file_name)
        print("Summary file name : " + file_name_in_summary)
        
        input_dataframe = pandas.read_csv(input_file)

        value = np.array(input_dataframe['value'])
        prediction = np.array(input_dataframe['prediction'])
        prediction_training = np.array(input_dataframe['prediction_training'])
        warp_distance = np.array(input_dataframe['warp_distance'])
        threshold_training = np.array(input_dataframe['threshold_training'])
        distance_threshold = np.array(input_dataframe['distance_threshold'])
        positive_detection = np.array(input_dataframe['positive_detection'])


        timestamp = np.array(input_dataframe['timestamp'])
        label = np.zeros(len(value))
        timestamp = helpers.string_to_date(timestamp, '%b %d, %Y')

        filler_string = []
        for i in range(0,len(label)):
            filler_string.append('')
        # timestamp = helpers.date_to_string(timestamp, '%Y-%m-%d')


        

        data = {'value':value,
                'prediction':prediction,
                'prediction_training':prediction_training,
                'label':label,
                'warp_distance':warp_distance,
                'threshold_training':threshold_training,
                'distance_threshold':distance_threshold,
                'positive_detection':positive_detection,
                'label_title':filler_string,
                'label_url':filler_string,
                'label_source':filler_string}
        out_dataframe = pandas.DataFrame(data, index=timestamp)
        out_dataframe.index.name = "timestamp"
        out_dataframe = out_dataframe[['value',
                                'prediction_training',
                                'prediction',
                                'label',
                                'warp_distance',
                                'threshold_training',
                                'distance_threshold',
                                'positive_detection',
                                'label_title',
                                'label_url',
                                'label_source']]
        
        
        # print( 'label_title ' + str(len(out_dataframe['label_title'])))
        # print( 'label_url ' + str(len(out_dataframe['label_url'])))
        # print( 'label_source ' + str(len(out_dataframe['label_source'])))

        news_timestamp = news_dataframe.index
        # print(news_timestamp)
        for i in news_timestamp:
            title = news_dataframe.at[i, 'title']
            if not isinstance(title, str):
                title = helpers.list_to_string(title, ';')
            url = news_dataframe.at[i, 'url']
            if not isinstance(url, str):
                url = helpers.list_to_string(url, ';')
            source = news_dataframe.at[i, 'source']
            if not isinstance(source, str):
                source = helpers.list_to_string(source, ';')
            
            for j in range(0,label_window_half):
                if i - timedelta(days=j) in out_dataframe.index :
                    out_dataframe.at[i - timedelta(days=j), 'label'] = 1.0
                    out_dataframe.at[i - timedelta(days=j), 'label_title'] = title
                    out_dataframe.at[i - timedelta(days=j), 'label_url'] = url
                    out_dataframe.at[i - timedelta(days=j), 'label_source'] = source
                    # print("label at " + str(i - timedelta(days=j)))
                if i + timedelta(days=j) in out_dataframe.index :
                    out_dataframe.at[i + timedelta(days=j), 'label'] = 1.0
                    out_dataframe.at[i + timedelta(days=j), 'label_title'] = title
                    out_dataframe.at[i + timedelta(days=j), 'label_url'] = url
                    out_dataframe.at[i + timedelta(days=j), 'label_source'] = source
                    # print("label at " + str(i + timedelta(days=j)))
            if not (i in out_dataframe.index) :
                news_dataframe.at[i, 'market_closed'] = 1.0
        # print(out_dataframe)

    

        # print( 'timestamp ' + str(len(timestamp)))
        # print( 'value ' + str(len(out_dataframe['value'])))
        # print( 'prediction ' + str(len(out_dataframe['prediction'])))
        # print( 'prediction_training ' + str(len(out_dataframe['prediction_training'])))
        # print( 'label ' + str(len(out_dataframe['label'])))
        # print( 'warp_distance ' + str(len(out_dataframe['warp_distance'])))
        # print( 'threshold_training ' + str(len(out_dataframe['threshold_training'])))
        # print( 'distance_threshold ' + str(len(out_dataframe['distance_threshold'])))
        # print( 'positive_detection ' + str(len(out_dataframe['positive_detection'])))
        # print( 'label_title ' + str(len(out_dataframe['label_title'])))
        # print( 'label_url ' + str(len(out_dataframe['label_url'])))
        # print( 'label_source ' + str(len(out_dataframe['label_source'])))



        # data = {'value':np.array(out_dataframe['value']),
        #         'prediction':np.array(out_dataframe['prediction']),
        #         'prediction_training':np.array(out_dataframe['prediction_training']),
        #         'label':np.array(out_dataframe['label']),
        #         'warp_distance':np.array(out_dataframe['warp_distance']),
        #         'threshold_training':np.array(out_dataframe['threshold_training']),
        #         'distance_threshold':np.array(out_dataframe['distance_threshold']),
        #         'positive_detection':np.array(out_dataframe['positive_detection']),
        #         'label_title':np.array(out_dataframe['label_title']),
        #         'label_url':np.array(out_dataframe['label_url']),
        #         'label_source':np.array(out_dataframe['label_source'])}
        # out_dataframe = pandas.DataFrame(data, index=timestamp)
        # out_dataframe.index.name = "timestamp"
        # out_dataframe = out_dataframe[['value',
        #                         'prediction_training',
        #                         'prediction',
        #                         'label',
        #                         'warp_distance',
        #                         'threshold_training',
        #                         'distance_threshold',
        #                         'positive_detection',
        #                         'label_title',
        #                         'label_url',
        #                         'label_source']]

        # print(out_dataframe['label'][tmp])
        # print(out_dataframe['label_title'][tmp])
        # print(out_dataframe['label_url'][tmp])
        # print(out_dataframe['label_source'][tmp])

        
        out_dataframe.to_csv(input_file)
        label = np.array(out_dataframe['label'])
        positive_detection = np.array(out_dataframe['positive_detection'])
        prediction_training = np.array(out_dataframe['prediction_training'])
        threshold_training = np.array(out_dataframe['threshold_training'])

        # Calculating confusion metrics
        metrics = confusion_metrics.confusion_metrics(label=label, positive_detection=positive_detection, prediction_training=prediction_training, threshold_training=threshold_training)
        metrics.calculate_metrics()

        model_dataframe.at[file_name, 'TP'] = metrics.get_TP()
        model_dataframe.at[file_name, 'TN'] = metrics.get_TN()
        model_dataframe.at[file_name, 'FP'] = metrics.get_FP()
        model_dataframe.at[file_name, 'FN'] = metrics.get_FN()

        print("##### ["+m+"] "+ str(count) + " CSV input File processed #####")
        count += 1
        if count == process_file_limit:
            break

    model_dataframe.to_csv(model_file_name)
    print("##### " + m + " done ! #####")
news_dataframe.to_csv(new_file)
print("All Done !")



