#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn
import warnings
import tensorflow as tf
import keras
import seaborn as sns
from pylab import *
from heapq import nlargest
from matplotlib import style
from sklearn import preprocessing, svm
from sklearn import linear_model
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout, Activation
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, \
    RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from keras import optimizers

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from pandas.plotting import register_matplotlib_converters
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None 

params = {
    "batch_size": 12, # frequency for weight updating
    "epochs": 100,
    "time_steps": 1, # number of units to go back in time
    "lr": 0.0010000,
    "fill_index": 3,
    "corr_thrs": 15
}

def get_headers(df):
    return(list(df.columns.values))

def read_data(path,index):
    fillers = def_fill()
    methods = fillers.values()

    if(isinstance(index, (int))):
        methd = fillers[index]
    else:
        methd = ''
    
    try:
        if(methd == 'ffill'):
            df = pd.read_csv(path,sep=',')
            df.fillna(method='ffill',inplace=True)
            return(df)
        elif(methd == 'bfill'):
            df = pd.read_csv(path,sep=',')
            df.fillna(method='bfill',inplace=True)
            return(df)
        elif(methd == 'mean'):
            df = pd.read_csv(path,sep=',')
            df.fillna(df.mean(),inplace=True)
            return(df)
        elif(methd == 'zero'):
            df = pd.read_csv(path,sep=',')
            df.fillna(0.0,inplace=True)
            return(df)       
        elif(methd not in methods):
            raise NameError
    except NameError:
        df = pd.read_csv(path,sep=',')
        # return by default the raw data (with missing data)
        return(df)
   
def view_data(df):
    print(df.head())
    
def def_fill():
    dct_fill = { 0: 'ffill',
                 1: 'bfill', 
                 2: 'mean' ,
                 3: 'zero' }
    return(dct_fill)

def scaler_def(index):
    if(index == 0):
        scl = StandardScaler()
    elif(index == 1):
        scl = MinMaxScaler()
    elif(index == 2):
        scl = MaxAbsScaler()
    elif(index == 3):
        scl = RobustScaler(quantile_range=(25, 75))
    elif(index == 4):
        scl = PowerTransformer(method='yeo-johnson')
    elif(index == 5):
        scl = PowerTransformer(method='box-cox')
    elif(index == 6):
        scl = QuantileTransformer(output_distribution='normal')
    elif(index == 7):
        scl = QuantileTransformer(output_distribution='uniform')
    elif(index == 8):
        scl = Normalizer()
    else:
        print('not a correct scaler defined')
    return(scl)

def get_optimizer(flag):
    if(flag=='rms'):
        return(optimizers.RMSprop(lr=params["lr"]))
    elif(flag=='sgd'):
        return(optimizers.SGD(lr=params["lr"], decay=1e-6, momentum=0.9, nesterov=True))
    elif(flag=='adam_tn'):
        return(optimizers.Adam(lr=params["lr"]))
    elif(flag==''):
        return('adam')

def model_build(df,headers,plotting):
    train_cols = headers
    df = df[train_cols]
    df_train, df_test = train_test_split(df, train_size=0.8, test_size=0.2, shuffle=False)
    # print("Train and Test size", len(df_train), len(df_test))

    # scale the feature MinMax, build array
    x = df_train.loc[:,train_cols].values
    scaler = scaler_def(1)
    x_train = scaler.fit_transform(x)
    x_test = scaler.transform(df_test.loc[:,train_cols])
    
    idx_out = len(train_cols)-1 # index of the output (here 'y')
    x_t, y_t = build_timeseries(x_train, idx_out)
    x_t, y_t = trim_dataset(x_t, params["batch_size"]), trim_dataset(y_t, params["batch_size"]) 
    x_temp, y_temp = build_timeseries(x_test, idx_out)
    x_val, x_test_t = np.split(trim_dataset(x_temp, params["batch_size"]),2)
    y_val, y_test_t = np.split(trim_dataset(y_temp, params["batch_size"]),2)
    
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(LSTM(128, batch_input_shape=(params["batch_size"], params["time_steps"], x_t.shape[2]),
                        dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=False,
                        kernel_initializer='random_uniform'))
    lstm_model.add(Dense(1,activation='tanh'))
    optmz = get_optimizer('')
    lstm_model.compile(loss='mse',optimizer=optmz)
    print(lstm_model.weights)

    print("######################")
    print(x_test_t.shape)
    print("######################")

    lstm_model.fit(x_t, y_t, epochs=params['epochs'], verbose=2, batch_size=params["batch_size"], \
                   shuffle=False, validation_data=(trim_dataset(x_val, params["batch_size"]), \
                trim_dataset(y_val, params["batch_size"])))

    # serialize model to JSON
    model_json = lstm_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    lstm_model.save_weights("model.h5")
    print("Saved model to disk")
    
    y_pred = lstm_model.predict(trim_dataset(x_test_t, params["batch_size"]), batch_size=params["batch_size"])
    y_pred = y_pred.flatten()
    y_test_t = trim_dataset(y_test_t, params["batch_size"])
    mse = mean_squared_error(y_test_t, y_pred)
    r2 = r2_score(y_test_t, y_pred)
    print("mse = ", mse)
    print("r2 = ", r2)  

    # convert the predicted value to range of real data
    y_pred_org = (y_pred * scaler.data_range_[idx_out]) + scaler.data_min_[idx_out]
    # scaler.inverse_transform(y_pred)
    y_test_t_org = (y_test_t * scaler.data_range_[idx_out]) + scaler.data_min_[idx_out]
    # scaler.inverse_transform(y_test_t)

    # Visualize the training data
    if(plotting):
        # plt.figure(figsize=(10.0,7.5))
        # plt.subplot(1, 2, 1)
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Test'], loc='upper left')

        plt.subplot(1,2,2)
        plt.plot(y_pred_org)
        plt.plot(y_test_t_org)
        plt.title('Prediction vs Real')
        plt.ylabel('Price')
        plt.xlabel('Days')
        plt.legend(['Prediction', 'Real'], loc='upper left')

        plt.show()
      
def build_timeseries(mat, y_col_index):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - params["time_steps"]
    dim_0 = mat.shape[0] - params["time_steps"]
    dim_1 = mat.shape[1]

    x = np.zeros((dim_0, params["time_steps"], dim_1))
    y = np.zeros((dim_0,))
    
    for i in range(dim_0):
        x[i] = mat[i:params["time_steps"]+i]
        y[i] = mat[params["time_steps"]+i, y_col_index]
    # print("length of time-series i/o",x.shape,y.shape)
    return x, y    

def trim_dataset(mat, batch_size):
    # trims dataset to a size that's divisible by params["batch_size"]
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

def get_cmtx_feat(df,corr_thrs,plotting):
    corr_matrix = df.corr()
    cmtx = corr_matrix["y"].sort_values(ascending=False)    
    dct = dict(zip(list(cmtx.keys()),cmtx.values))
    dct_select = dict((k, v) for k, v in dct.items() if v >= float(corr_thrs)/100.0 and k != 'y')

    if(plotting):
        s=sns.heatmap(corr_matrix,cmap='coolwarm') 
        s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
        s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)
        plt.show()

    return(dct_select)
       
        
if __name__ == '__main__':
    dirc = os.path.join(os.getcwd(),'data_sample_1e4.csv')
    df_fill = read_data(dirc,params["fill_index"])

    # corr_thrs = params["corr_thrs"]
    # cm = get_cmtx_feat(df_fill,corr_thrs,False)
    # top_features = nlargest(len(cm),cm,key=cm.get)

    top_features = get_headers(df_fill)
    model_build(df_fill,top_features,True)
    
