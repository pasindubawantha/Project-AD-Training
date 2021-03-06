from . import model_helpers
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

class arma:
    def __init__(self, data, training_ratio, ar_max=3, ma_max=3):
        self.ar_max = ar_max # maximum AR parameter
        self.ma_max = ma_max # maximum MA parameter
        training_data_end = int(len(data)*training_ratio)
        testing_data_start = training_data_end
        self.training_data = data[:training_data_end].astype('float64')
        self.testing_data = data[testing_data_start:].astype('float64')
        self.params = None

    def train(self):
        ar_max = self.ar_max
        ma_max = self.ma_max
        params = np.zeros((ar_max+1, ma_max+1))
        y = self.training_data

        for ar in range(0,ar_max+1):
            for ma in range(0,ma_max+1):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        arma_model = sm.tsa.ARMA(y, order=(ar,ma))
                        arma_result = arma_model.fit(trend='c', disp=-1)
                    y_prediction = arma_result.predict()
                    mse = model_helpers.MSE(y,y_prediction)
                except ValueError:
                    mse = float("inf") 
                except np.linalg.LinAlgError:
                    mes = float("inf") 
                params[ar][ma] = mse
                print("("+str(ar)+","+str(ma)+") : " +str(mse))
        
        minAR, minMA = model_helpers.get_min_matrix(params)
        print("Best model : ("+str(minAR)+","+str(minMA)+") | with MSE = "+str(params[minAR][minMA]))
        self.params = params
        self.ar = minAR
        self.ma = minMA
        return minAR,minMA
    
    def get_output(self):
        ar = self.ar
        ma = self.ma
        y = self.testing_data
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                arma_model = sm.tsa.ARMA(y, order=(ar,ma))
                arma_result = arma_model.fit(trend='c', disp=-1)
            except ValueError:
                print("model : ("+str(ar)+","+str(ma)+") failed ! getting next best model")
                self.params[ar][ma] = float("inf")
                minAR, minMA = model_helpers.get_min_matrix(self.params)
                print("Best model : ("+str(minAR)+","+str(minMA)+") | with MSE = "+str(self.params[minAR][minMA]))
                self.ar = minAR
                self.ma = minMA
                return self.get_output()
            except np.linalg.LinAlgError:
                print("model : ("+str(ar)+","+str(ma)+") failed ! getting next best model")
                self.params[ar][ma] = float("inf")
                minAR, minMA = model_helpers.get_min_matrix(self.params)
                print("Best model : ("+str(minAR)+","+str(minMA)+") | with MSE = "+str(self.params[minAR][minMA]))
                self.ar = minAR
                self.ma = minMA
                return self.get_output()
                
        y_prediction = arma_result.predict()
        return ar, ma, np.array(y_prediction)