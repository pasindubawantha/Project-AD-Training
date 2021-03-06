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


# args
input_directory = "../results/data"
input_summary_file = "../data/lse_summary.csv"
output_directory = "../results"

max_training_ratio = 0.15
# threshold_training_ratio = 0.25
# prediction_training_ratio = 0.75
max_training_ratio_buffer = 0.95
threshold_max_multipler = 2

models = ["arma","arima","lstm","cnn","lstmcnn_wsum_combination","lstmcnn_wsum_layer_lstmDlayer","lstmcnn_wsum_layer"]
# models = ["arma","lstm","cnn","lstmcnn_wsum_combination","lstmcnn_wsum_layer_lstmDlayer","lstmcnn_wsum_layer"]
# models = ["lstmcnn_kerascombinantion_vanila"]
# models = ["lstmcnn_kerascombinantion"]
# models = ["lstmcnn_kerascombinantion_vanila"]
# models = ["lstmcnn_wsum_layer"]


input_summary = pandas.read_csv(input_summary_file, index_col="file")

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
    
    mse = np.array(model_dataframe['mse'])
    filler_values = np.zeros(len(mse))
    filler_string = []
    for i in range(len(mse)):
            filler_string.append('')

    data = {'mse':model_dataframe['mse'], 
            'TP':filler_values,
            'FP':filler_values,
            'FN':filler_values,
            'TN':filler_values,
            'parameters':model_dataframe['parameters'],
            'threshold_parameters':filler_string,
            'no_of_anomalies':filler_values,
            'first_label':filler_values,
            'length':filler_values,
            "first_label_ratio":filler_values}
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
        # if "Volume" in input_file:
        #         continue
        print("Processing ["+m+"]" + input_file)
        file_name = input_file.split("/")[-1]
        file_name_in_summary = file_name.split(".metric-")[0] + ".csv"
        print("File name : " + file_name)
        print("Summary file name : " + file_name_in_summary)
        
        input_dataframe = pandas.read_csv(input_file)
        value = np.array(input_dataframe['value'])
        prediction = np.array(input_dataframe['prediction'])
        label = np.array(input_dataframe['label'])
        prediction_training = np.array(input_dataframe['prediction_training'])


        jsonf_name = input_file[:-3] + 'json'
        jsonf = open(jsonf_name, "r")
        jsond = json.load(jsonf)

        dtw_window_size = int(jsond['dtw_window'])

        prediction_training_stops = 0
        for i in range(0, len(prediction_training)):
                if prediction_training[i] == 0:
                        break
                prediction_training_stops += 1

        testing_value = value[prediction_training_stops:]
        testing_prediction = prediction[prediction_training_stops:]
        
        ## Training ratio
        first_label_ratio = input_summary['first_label_ratio'][file_name_in_summary]
        if first_label_ratio < max_training_ratio:
            total_training_length = int(first_label_ratio * max_training_ratio_buffer * len(value))
        else:
            total_training_length = (max_training_ratio * len(value))
        
        threshold_training_count = total_training_length - prediction_training_stops

        if threshold_training_count <= 0:
                print("Cant train threshold !! (lstmcnn input window)sequance length is too large")
                model_dataframe.at[file_name, 'threshold_parameters'] = "Cant train threshold !! (lstmcnn input window)sequance length is too large"
                continue

        training_ratio = float(threshold_training_count)/float(len(testing_value))

        detector_instance = detector.detector(values=testing_value, predictions=testing_prediction)
        warp_distance = detector_instance.calculate_distances(comparision_window_size=dtw_window_size)
        threshold = detector_instance.set_threshold(training_ratio=training_ratio, max_multipler=threshold_max_multipler)
        positive_detection = detector_instance.get_anomalies()
        threshold_training_starts = prediction_training_stops
        threshold_training_size = int(len(testing_value)*training_ratio)
        
        threshold_ignore = np.zeros(threshold_training_starts)
        threshold_training = np.ones(threshold_training_size)
        threshold_testing = np.zeros(len(value)-threshold_training_starts - threshold_training_size)
        threshold_training_colomn = np.append(threshold_ignore, threshold_training)
        threshold_training_colomn = np.append(threshold_training_colomn, threshold_testing)
        
        # for i in range(threshold_training_size):


        threshold = np.ones(len(warp_distance) - threshold_training_size)*threshold

        threshold = np.append(np.ones(len(threshold_ignore) + threshold_training_size)*(-1), threshold)
        warp_distance = np.append(threshold_ignore, warp_distance)
        positive_detection = np.append(threshold_ignore, positive_detection)

        data = {'value':value,
                'prediction':prediction,
                'prediction_training':prediction_training,
                'label':label,
                'warp_distance':warp_distance,
                'threshold_training':threshold_training_colomn,
                'distance_threshold':threshold,
                'positive_detection':positive_detection,
                # 'label_title':input_dataframe['label_title'],
                # 'label_url':input_dataframe['label_url'],
                # 'label_source':input_dataframe['label_source'],
                }
        out_dataframe = pandas.DataFrame(data, index=np.array(input_dataframe['timestamp']))
        out_dataframe.index.name = "timestamp"
        out_dataframe = out_dataframe[['value',
                                'prediction_training',
                                'prediction',
                                'label',
                                'warp_distance',
                                'threshold_training',
                                'distance_threshold',
                                'positive_detection'
                                # 'label_title',
                                # 'label_url',
                                # 'label_source'
                                ]]
        out_dataframe.to_csv(input_file)


        # Calculating confusion metrics
        metrics = confusion_metrics.confusion_metrics(label=label, positive_detection=positive_detection, prediction_training=prediction_training, threshold_training=threshold_training_colomn)
        metrics.calculate_metrics()

        model_dataframe.at[file_name, 'TP'] = metrics.get_TP()
        model_dataframe.at[file_name, 'TN'] = metrics.get_TN()
        model_dataframe.at[file_name, 'FP'] = metrics.get_FP()
        model_dataframe.at[file_name, 'FN'] = metrics.get_FN()
        model_dataframe.at[file_name, 'threshold_parameters'] = "comparision_window_size="+str(dtw_window_size)+";threshold_max_multipler="+str(threshold_max_multipler)+";training_ratio="+str(training_ratio)+";"
        model_dataframe.at[file_name, 'no_of_anomalies'] = input_summary['no_of_anomalies'][file_name_in_summary]
        model_dataframe.at[file_name, 'first_label'] = input_summary['first_label'][file_name_in_summary]
        model_dataframe.at[file_name, 'length'] = input_summary['length'][file_name_in_summary]
        model_dataframe.at[file_name, 'first_label_ratio'] = input_summary['first_label_ratio'][file_name_in_summary]

        print("##### ["+m+"] "+ str(count) + " CSV input File processed #####")
        count += 1

    model_dataframe.to_csv(model_file_name)
    print("##### " + m + " done ! #####")
print("All Done !")



