import json
import pandas
import math
import sys
import os
import numpy as np
import re
import shutil
import helpers
import models.arma as arma
import models.arima as arima
import models.lstm as lstm
import models.cnn as cnn
import models.lstmcnn_wsum_combination as lstmcnn_wsum_combination
import models.lstmcnn_wsum_layer_lstmDlayer as lstmcnn_wsum_layer_lstmDlayer
import models.lstmcnn_wsum_layer as lstmcnn_wsum_layer

# args
input_directory = "../data/lseg"
# input_file_metrics = ["Vol","Close","Open","High","Low"]
input_file_metrics = ["Val"]

input_summary_file = "../data/lse_summary.csv"
output_directory = "../results"

max_training_ratio = 0.15
prediction_training_ratio = 0.75
max_training_ratio_buffer = 0.95

models = ["arma","arima","lstm","cnn","lstmcnn_wsum_combination","lstmcnn_wsum_layer_lstmDlayer","lstmcnn_wsum_layer"]
# models = ["lstmcnn_wsum_combination","lstmcnn_wsum_layer_lstmDlayer","lstmcnn_wsum_layer"]
# models = ["lstm","cnn","lstmcnn_wsum_layer"]
# models = ["lstmcnn_wsum_layer"]
# models = ["lstmcnn_wsum_layer_lstmDlayer"]
# models = ["lstmcnn_wsum_layer"]





# get all csv files in input directory
reg_x = re.compile(r'\.(csv)')
csv_input_files = []
for path, dnames, fnames in os.walk(input_directory):
    csv_input_files.extend([os.path.join(path, f) for f in fnames if reg_x.search(f)])

csv_input_files.sort()

input_summary = pandas.read_csv(input_summary_file, index_col="file")

try:
    shutil.rmtree(output_directory + "/data")
except OSError:
    print("No previous ", output_directory + "/data")

os.mkdir(output_directory + "/data")

for m in models:
    os.mkdir(output_directory + "/data/" + m)
    output_files = []
    output_files.append("file,mse,parameters")
   
    helpers.dump_results(output_files, output_directory, m)

    print("##### ["+m+"]"+ str(len(csv_input_files)) + " CSV input files to process #####")
    count = 1
    for f in csv_input_files:
        print("Processing ["+m+"]" + f)
        
        jsonf_name = f[:-3] + 'json'
        jsonf = open(jsonf_name, "r")
        jsond = json.load(jsonf)

        epochs = int(jsond['prediction_model']['trainingIterations'])
        learningRate = float(jsond['prediction_model']['learningRate'])
        CL1strides = int(jsond['prediction_model']['model']['CNN']['ConvolutionLayers'][0]['stride'])
        print("")
        CL1kernal_size = int(jsond['prediction_model']['model']['CNN']['ConvolutionLayers'][0]['filterSize'])
        CL1filters = int(jsond['prediction_model']['model']['CNN']['ConvolutionLayers'][0]['filters'])

        PL1pool_size = 1

        DL1units = int(jsond['prediction_model']['model']['CNN']['FullyConnectedLayers'][0]['outputs'])
        DL2units = int(jsond['prediction_model']['model']['CNN']['FullyConnectedLayers'][1]['outputs'])
        DL3units = int(jsond['prediction_model']['model']['CNN']['FullyConnectedLayers'][2]['outputs'])

        sequance_length = int(jsond['prediction_model']['model']['CNN']['matWidth']) * int(jsond['prediction_model']['model']['CNN']['matHeight'])

        lstmCells = int(jsond['prediction_model']['model']['LSTM']['memCells'])
        cnnWeight = float(jsond['prediction_model']['model']['cnnW'])
        lstmWeight = float(jsond['prediction_model']['model']['lstmW'])

        batch_size = 1

        dataframe = pandas.read_csv(f)
        timestamp = np.array(dataframe['timestamp'])

        for metric in input_file_metrics:
            print("Processing ["+m+"]" + f + " Metric : " + metric)
            value = helpers.remove_commas(np.array(dataframe[metric])).astype(np.float)

            label = np.array(dataframe['label'])
            # label = np.zeros(len(value))


            ## Training ratio
            fname = f.split("/")[-1]
            first_label_ratio = input_summary['first_label_ratio'][fname]
            if first_label_ratio < max_training_ratio:
                training_ratio = first_label_ratio*max_training_ratio_buffer*prediction_training_ratio
            else:
                training_ratio = max_training_ratio*prediction_training_ratio

            if int(training_ratio*len(value)) +3 < sequance_length:
                training_ratio = float(sequance_length + 3) / float(len(value))

            # running model
            if(m=="arma"):
                #parms
                ar_max = 4
                ma_max = 4

                arma_model = arma.arma(data=value, training_ratio=training_ratio, ar_max=ar_max, ma_max=ma_max)
                arma_model.train()
                ar, ma, prediction = arma_model.get_output()
                params = "ar="+str(ar)+";ma="+str(ma)+";training_ratio="+str(training_ratio)
            elif(m=="arima"):
                #params
                ar_max = 4
                d_max = 2
                ma_max = 4

                arima_model = arima.arima(data=value, training_ratio=training_ratio, ar_max=ar_max, d_max=d_max, ma_max=ma_max)
                arima_model.train()
                
                ar,d,ma,prediction = arima_model.get_output()
                params = "ar="+str(ar)+";d="+str(d)+";ma="+str(ma)+";training_ratio="+str(training_ratio)
            elif(m=="lstm"):
                #params
                # lstmCells = 10

                lstm_model = lstm.lstm(data=value,  epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length, lstmCells=lstmCells, learningRate=learningRate)
                lstm_model.train()
                params = "lstmCells="+str(lstmCells)+";learningRate="+str(learningRate)+";epochs="+str(epochs)+";batch_size="+str(batch_size)+";training_ratio="+str(training_ratio)+";sequance_length="+str(sequance_length)
                prediction = lstm_model.get_output()
            elif(m=="cnn"):
                #params
                # CL1filters = 1
                # CL1kernal_size = 2
                # CL1strides = 1
                # PL1pool_size = 1
                # DL1units = 20
                # DL2units = 5
                # DL3units = 1

                cnn_model = cnn.cnn(data=value, epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length,CL1filters=CL1filters, CL1kernal_size=CL1kernal_size, CL1strides=CL1strides, PL1pool_size=PL1pool_size, DL1units=DL1units, DL2units=DL2units, DL3units=DL3units, learningRate=learningRate)
                cnn_model.train()
                params = "CL1filters="+str(CL1filters)+";CL1kernal_size="+str(CL1kernal_size)+";CL1strides="+str(CL1strides)+";PL1pool_size="+str(PL1pool_size)+";DL1units="+str(DL1units)+";DL2units="+str(DL2units)+";DL3units="+str(DL3units)+";epochs="+str(epochs)+";batch_size="+str(batch_size)+";training_ratio="+str(training_ratio)+";sequance_length="+str(sequance_length)+";learningRate="+str(learningRate)

                prediction = cnn_model.get_output()
            elif(m=="lstmcnn_wsum_combination"):
                #params
                # lstmWeight = 0.5
                # cnnWeight = 0.5
                #lstm params
                # lstmCells=10
                #cnn params
                # CL1filters = 1
                # CL1kernal_size = 2
                # CL1strides = 1
                # PL1pool_size = 1
                CNNDL1units = DL1units
                CNNDL2units = DL2units
                CNNDL3units = DL3units
                LSTMDL1units = DL1units
                LSTMDL2units = DL2units
                LSTMDL3units = DL3units

                lstmcnn_wsum_combination_model = lstmcnn_wsum_combination.lstmcnn_wsum_combination(data=value, epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length, lstmCells=lstmCells, LSTMDL1units=LSTMDL1units, LSTMDL2units=LSTMDL2units, LSTMDL3units=LSTMDL3units, CL1filters=CL1filters, CL1kernal_size=CL1kernal_size, CL1strides=CL1strides, PL1pool_size=PL1pool_size, CNNDL1units=CNNDL1units, CNNDL2units=CNNDL2units, CNNDL3units=CNNDL3units, lstmWeight=lstmWeight, cnnWeight=cnnWeight, learningRate=learningRate)
                lstmcnn_wsum_combination_model.train()
                params = "lstmWeight="+str(lstmWeight)+";cnnWeight="+str(cnnWeight)+";lstmCells="+str(lstmCells)+";LSTMDL1units="+str(LSTMDL1units)+";LSTML2units="+str(LSTMDL2units)+";LSTMDL3units="+str(LSTMDL3units)+";CL1filters="+str(CL1filters)+";CL1kernal_size="+str(CL1kernal_size)+";CL1strides="+str(CL1strides)+";PL1pool_size="+str(PL1pool_size)+";CNNDL1units="+str(CNNDL1units)+";CNNDL2units="+str(CNNDL2units)+";CNNDL3units="+str(CNNDL3units)+";epochs="+str(epochs)+";batch_size="+str(batch_size)+";training_ratio="+str(training_ratio)+";sequance_length="+str(sequance_length)+";learningRate="+str(learningRate)

                prediction = lstmcnn_wsum_combination_model.get_output()
            elif(m=="lstmcnn_wsum_layer_lstmDlayer"):
                #params
                # lstmWeight = 0.5
                # cnnWeight = 0.5
                #lstm params
                # lstmCells=10
                #cnn params
                # CL1filters = 1
                # CL1kernal_size = 2
                # CL1strides = 1
                # PL1pool_size = 1
                CNNDL1units = DL1units
                CNNDL2units = DL2units
                CNNDL3units = DL3units
                LSTMDL1units = DL1units
                LSTMDL2units = DL2units
                LSTMDL3units = DL3units

                lstmcnn_wsum_layer_lstmDlayer_model = lstmcnn_wsum_layer_lstmDlayer.lstmcnn_wsum_layer_lstmDlayer(data=value, epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length, lstmCells=lstmCells, LSTMDL1units=LSTMDL1units, LSTMDL2units=LSTMDL2units, LSTMDL3units=LSTMDL3units, CL1filters=CL1filters, CL1kernal_size=CL1kernal_size, CL1strides=CL1strides, PL1pool_size=PL1pool_size, CNNDL1units=CNNDL1units, CNNDL2units=CNNDL2units, CNNDL3units=CNNDL3units, lstmWeight=lstmWeight, cnnWeight=cnnWeight, learningRate=learningRate)
                lstmcnn_wsum_layer_lstmDlayer_model.train()
                params = "lstmWeight="+str(lstmWeight)+";cnnWeight="+str(cnnWeight)+";lstmCells="+str(lstmCells)+";LSTMDL1units="+str(LSTMDL1units)+";LSTML2units="+str(LSTMDL2units)+";LSTMDL3units="+str(LSTMDL3units)+";CL1filters="+str(CL1filters)+";CL1kernal_size="+str(CL1kernal_size)+";CL1strides="+str(CL1strides)+";PL1pool_size="+str(PL1pool_size)+";CNNDL1units="+str(CNNDL1units)+";CNNDL2units="+str(CNNDL2units)+";CNNDL3units="+str(CNNDL3units)+";epochs="+str(epochs)+";batch_size="+str(batch_size)+";training_ratio="+str(training_ratio)+";sequance_length="+str(sequance_length)+";learningRate="+str(learningRate)

                prediction = lstmcnn_wsum_layer_lstmDlayer_model.get_output()
            elif(m=="lstmcnn_wsum_layer"):
                #params
                # lstmWeight = 0.5
                # cnnWeight = 0.5
                #lstm params
                # lstmCells=10
                #cnn params
                # CL1filters = 1
                # CL1kernal_size = 2
                # CL1strides = 1
                # PL1pool_size = 1
                CNNDL1units = DL1units
                CNNDL2units = DL2units
                CNNDL3units = DL3units

                lstmcnn_wsum_layer_model = lstmcnn_wsum_layer.lstmcnn_wsum_layer(data=value, epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length, lstmCells=lstmCells, CL1filters=CL1filters, CL1kernal_size=CL1kernal_size, CL1strides=CL1strides, PL1pool_size=PL1pool_size, CNNDL1units=CNNDL1units, CNNDL2units=CNNDL2units, CNNDL3units=CNNDL3units, lstmWeight=lstmWeight, cnnWeight=cnnWeight, learningRate=learningRate)

                lstmcnn_wsum_layer_model.train()

                params = "lstmWeight="+str(lstmWeight)+";cnnWeight="+str(cnnWeight)+";lstmCells="+str(lstmCells)+";CL1filters="+str(CL1filters)+";CL1kernal_size="+str(CL1kernal_size)+";CL1strides="+str(CL1strides)+";PL1pool_size="+str(PL1pool_size)+";CNNDL1units="+str(CNNDL1units)+";CNNDL2units="+str(CNNDL2units)+";CNNDL3units="+str(CNNDL3units)+";epochs="+str(epochs)+";batch_size="+str(batch_size)+";training_ratio="+str(training_ratio)+";sequance_length="+str(sequance_length)+";learningRate="+str(learningRate)

                prediction = lstmcnn_wsum_layer_model.get_output()
            else:
                print("Invalid Model!")

            testing_start = int(training_ratio*len(value))
            training_colomn = np.append(np.ones(testing_start), np.zeros(len(value)-testing_start))

            testing_prediction = prediction
            prediction = np.append(np.zeros(testing_start), prediction)



            # label_title = np.array(dataframe['label_title'])
            # label_url = np.array(dataframe['label_url'])
            # label_source = np.array(dataframe['label_source'])

            data = {'prediction':prediction, 
                    'value':value,
                    'prediction_training':training_colomn,
                    'label':label
                    # 'label_title':label_title,
                    # 'label_url':label_url,
                    # 'label_source':label_source 
                    } 
            dataframe_out = pandas.DataFrame(data, index=timestamp)
            dataframe_out.index.name = "timestamp"
            dataframe_out = dataframe_out[['value',
                'prediction_training',
                'prediction',
                'label'
                # 'label_title',
                # 'label_url',
                # 'label_source'
                ]]
            out_file = helpers.get_result_file_name(f, output_directory, m)
            out_file_name = out_file[:-4] + ".metric-" + metric + ".csv"
            dataframe_out.to_csv(out_file_name)
            new_jsonf_name = out_file_name[:-3] + "json"
            shutil.copyfile(jsonf_name, new_jsonf_name)

            testing_value = value[testing_start:]
            mse = helpers.MSE(testing_value, testing_prediction)
            output_files.append( helpers.get_result_dump_name(out_file_name) + "," + str(mse) + "," + params )

            helpers.dump_results(output_files, output_directory, m)
        print("##### ["+m+"]"+ str(count) + " CSV input File processed #####")
        count += 1

    print("##### " + m + " done ! #####")
print("All Done !")