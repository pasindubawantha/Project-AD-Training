import pandas
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, Flatten
from keras.utils import plot_model
from keras import optimizers

# f = "../../data/artificialNoAnomaly/art_daily_no_noise.csv"
# dataFrame = pandas.read_csv(f)
# data = np.array(dataFrame['value'])
# print(data)


class cnn:
    def __init__(self, data, epochs, batch_size, training_ratio, sequance_length,CL1filters=1, CL1kernal_size=2, CL1strides=1, PL1pool_size=1, DL1units=20, DL2units=5, DL3units=1, learningRate=0.001):
        self.CL1filters = CL1filters
        self.CL1kernal_size = CL1kernal_size
        self.CL1strides = CL1strides
        self.sequance_length = sequance_length
        self.PL1pool_size = PL1pool_size
        self.DL1units = DL1units
        self.DL2units = DL2units
        self.DL3units = DL3units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learningRate = learningRate
        training_data_end = int(training_ratio*len(data))
        testing_data_start = training_data_end - sequance_length
        training_data = data[:training_data_end]
        testing_data = data[testing_data_start:]

        self.training_feature_set = []
        self.labels = []
        for i in range(sequance_length, len(training_data)):
            self.training_feature_set.append(training_data[i-sequance_length:i])
            self.labels.append(training_data[i])
        self.labels = np.array(self.labels)
        self.training_feature_set = np.array(self.training_feature_set)
        self.training_feature_set = np.reshape(self.training_feature_set, (self.training_feature_set.shape[0], self.training_feature_set.shape[1], 1))
        
        self.testing_feature_set = []
        for i in range(sequance_length, len(testing_data)):
            self.testing_feature_set.append(testing_data[i-sequance_length:i])
        self.testing_feature_set = np.array(self.testing_feature_set)
        self.testing_feature_set = np.reshape(self.testing_feature_set, (self.testing_feature_set.shape[0], self.testing_feature_set.shape[1], 1))

    def train(self):
        self.model = Sequential()
        self.model.add(Conv1D(filters=self.CL1filters, kernel_size=self.CL1kernal_size,strides=self.CL1strides, input_shape=(self.sequance_length, 1)))
        self.model.add(MaxPooling1D(pool_size=self.PL1pool_size))
        self.model.add(Flatten())
        self.model.add(Dense(units = self.DL1units))
        self.model.add(Dense(units = self.DL2units))
        self.model.add(Dense(units = self.DL3units))

        adam = optimizers.Adam(lr=self.learningRate)

        self.model.compile(optimizer = adam, loss = 'mean_squared_error', metrics=["mse"])
        self.model.fit(self.training_feature_set, self.labels, batch_size=self.batch_size, epochs=self.epochs)
    
    def get_output(self):
        predictions = self.model.predict(self.testing_feature_set)
        ret_prediction = np.zeros(predictions.shape[0])
        for i in range(0, predictions.shape[0]):
            ret_prediction[i] = predictions[i][0]
        return ret_prediction

    def print_model(self,f):
        plot_model(self.model, to_file=f)