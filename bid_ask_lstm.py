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
    "corr_thrs": 10
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

def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def get_optimizer(flag):
    if(flag=='rms'):
        return(optimizers.RMSprop(lr=params["lr"]))
    elif(flag=='sgd'):
        return(optimizers.SGD(lr=params["lr"], decay=1e-6, momentum=0.9, nesterov=True))
    elif(flag=='adam_tn'):
        return(optimizers.Adam(lr=params["lr"]))
    elif(flag==''):
        return('adam')
      
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

def build_model(X,y):
    lstm_model = Sequential()
    # (batch_size, timesteps, data_dim)
    lstm_model.add(LSTM(128, input_shape=(params["time_steps"], X.shape[2]),
                        dropout=0.0, recurrent_dropout=0.0, stateful=False, return_sequences=False,
                        kernel_initializer='random_uniform'))
    lstm_model.add(Dense(1,activation='tanh'))
    optmz = get_optimizer('adam_tn')
    lstm_model.compile(loss='mse',optimizer=optmz)
    lstm_model.fit(X,y, epochs=params['epochs'], verbose=0)

    # serialize model to JSON
    # model_json = lstm_model.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    #     # serialize weights to HDF5
    # lstm_model.save_weights("model.h5")
    # print("Saved model to disk")


    # lstm_model.history['loss']
    # lstm_model.history['val_loss']

    # history = lstm_model.fit(x_t, y_t, epochs=params['epochs'], verbose=2, batch_size=params["batch_size"], \
    #                          shuffle=False, validation_data=(trim_dataset(x_val, params["batch_size"]), \
    #                         trim_dataset(y_val, params["batch_size"])))
    return(lstm_model)

if __name__ == '__main__':
    dirc = os.path.join(os.getcwd(),'data_sample_1e2.csv')
    df_fill = read_data(dirc,params["fill_index"])
    df_len = len(df_fill)
    
    # corr_thrs = params["corr_thrs"]
    # cm = get_cmtx_feat(df_fill,corr_thrs,False)
    # top_features = nlargest(len(cm),cm,key=cm.get)
    # top_features = top_features + ['y']
    
    top_features = get_headers(df_fill)
    
    df_fill = df_fill.loc[:,top_features]
    scaler = scaler_def(1)
    df_fill[top_features] = scaler.fit_transform(df_fill[top_features])
    
    err2_tally = y2_tally = 0
    
    out = pd.DataFrame(df_fill.iloc[0]).T
    for el in range(1,len(df_fill)-1):
        n_steps = params["time_steps"]
        out = pd.concat([out,pd.DataFrame(df_fill.iloc[el]).T],axis=0)
        dataset = out.values
        X, y = split_sequences(dataset, n_steps)

        n_features = X.shape[2]
        
        # model (build & fit)
        lstm_model = build_model(X,y)
      
        # demonstrate prediction
        x_input = np.array(df_fill.iloc[el+1][:-1].T.values)
        x_input = x_input.reshape((1, n_steps, n_features))

        yexact_scaled = np.array(df_fill.iloc[el+1][-1].T)
        ypredt_scaled = lstm_model.predict(x_input, verbose=0).flatten()[0]
        
        yexact = (yexact_scaled * scaler.data_range_[-1]) + scaler.data_min_[-1]
        ypredt = ((ypredt_scaled * scaler.data_range_[-1]) + scaler.data_min_[-1]).flatten()[0]
        
        err2_tally += (yexact - ypredt) ** 2
        y2_tally += yexact ** 2

        txt = "{} {} \n".format(ypredt,yexact) 

        with open("out.txt", "a") as myfile:
            myfile.write(txt)
        print(txt)
    
r2 = 1.0 - err2_tally / y2_tally
print(r2)
