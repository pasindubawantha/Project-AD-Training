import models.lstm as lstm
import models.cnn as cnn
import models.lstmcnn_wsum_combination as lstmcnn_wsum_combination
import models.lstmcnn_wsum_layer_lstmDlayer as lstmcnn_wsum_layer_lstmDlayer
import models.lstmcnn_wsum_layer as lstmcnn_wsum_layer
import pandas
import numpy as np
import helpers


f = "../data/lseg/LSE Historical Data 2009.csv"
dataframe = pandas.read_csv(f)
value = helpers.remove_commas(np.array(dataframe['Price'])).astype(np.float)

training_ratio = 0.1
sequance_length = 20
epochs = 1
batch_size = 100
learningRate = 0.001


lstmCells = 10

lstm_model = lstm.lstm(data=value,  epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length, lstmCells=lstmCells)
lstm_model.train()
lstm_model.print_model("../results/resources/img/lstm_model.png")


CL1filters = 1
CL1kernal_size = 2
CL1strides = 1
PL1pool_size = 1
DL1units = 20
DL2units = 5
DL3units = 1

cnn_model = cnn.cnn(data=value, epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length,CL1filters=CL1filters, CL1kernal_size=CL1kernal_size, CL1strides=CL1strides, PL1pool_size=PL1pool_size, DL1units=DL1units, DL2units=DL2units, DL3units=DL3units)
cnn_model.train()
cnn_model.print_model("../results/resources/img/cnn_model.png")


#params
lstmWeight = 0.5
cnnWeight = 0.5
#lstm params
lstmCells=10
#cnn params
CL1filters = 1
CL1kernal_size = 2
CL1strides = 1
PL1pool_size = 1
CNNDL1units = 20
CNNDL2units = 5
CNNDL3units = 1
LSTMDL1units = 20
LSTMDL2units = 5
LSTMDL3units = 1

lstmcnn_wsum_layer_lstmDlayer_model = lstmcnn_wsum_layer_lstmDlayer.lstmcnn_wsum_layer_lstmDlayer(data=value, epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length, lstmCells=lstmCells, LSTMDL1units=LSTMDL1units, LSTMDL2units=LSTMDL2units, LSTMDL3units=LSTMDL3units, CL1filters=CL1filters, CL1kernal_size=CL1kernal_size, CL1strides=CL1strides, PL1pool_size=PL1pool_size, CNNDL1units=CNNDL1units, CNNDL2units=CNNDL2units, CNNDL3units=CNNDL3units, lstmWeight=lstmWeight, cnnWeight=cnnWeight)
lstmcnn_wsum_layer_lstmDlayer_model.train()
lstmcnn_wsum_layer_lstmDlayer_model.print_model("../results/resources/img/lstmcnn_wsum_layer_lstmDlayer_model.png")

lstmcnn_wsum_layer_model = lstmcnn_wsum_layer.lstmcnn_wsum_layer(data=value, epochs=epochs, batch_size=batch_size, training_ratio=training_ratio, sequance_length=sequance_length, lstmCells=lstmCells, CL1filters=CL1filters, CL1kernal_size=CL1kernal_size, CL1strides=CL1strides, PL1pool_size=PL1pool_size, CNNDL1units=CNNDL1units, CNNDL2units=CNNDL2units, CNNDL3units=CNNDL3units, lstmWeight=lstmWeight, cnnWeight=cnnWeight, learningRate=learningRate)
lstmcnn_wsum_layer_model.train()
lstmcnn_wsum_layer_model.print_model("../results/resources/img/lstmcnn_wsum_layer_model.png")
