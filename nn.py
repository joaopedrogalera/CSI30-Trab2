import csv
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, utils
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

class NN:
    def __init__(self, log=False):
        self.log = log
        
        self.model = Sequential([
                Dense(256, activation='relu'),
                Dense(256, activation='relu'),
                Dense(128, activation='relu'),
                Dense(1)
        ])
        
        self.model.compile(
            loss=self.__rmse,
            optimizer=Adam(),
            metrics=[self.__rmse]
        )
        
    def __rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
        
    def Train(self, data_treino, classes_treino):
        data = np.array(data_treino)
        classes = np.array(classes_treino)
        
        training = self.model.fit(x=data, y=classes, epochs=400, verbose=self.log)
        
        return training
        
    def Predict(self, X):
        X_test = np.array(X)
        
        return self.model.predict(X_test)
        
        