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
from sklearn.metrics import accuracy_score

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from pandas.plotting import register_matplotlib_converters
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None 

with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer


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

def model_build(df,headers):
    train_cols = headers
    df = df[train_cols]
    
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values

    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

    model = Sequential()
    model.add(Dense(32, input_dim=60, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    
    optmz = get_optimizer('')
    model.compile(loss='mse',optimizer=optmz)
    model.fit(X_train,y_train,epochs=100,batch_size=50,verbose=0)
  
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test,y_pred)
    print(accuracy)

    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")

def get_history(df,headers):
    train_cols = headers
    df = df[train_cols]
    
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1].values

    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

    model = Sequential()
    model.add(Dense(32, input_dim=60, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    
    optmz = get_optimizer('')
    model.compile(loss='mse',optimizer=optmz,metrics=['accuracy'])
    history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64, verbose=0)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.show()
    
    y_pred = model.predict(X_test)
    accuracy = r2_score(y_test,y_pred)
    print(accuracy)
    
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
    dirc = os.path.join(os.getcwd(),TO_REPLACE)
    df_fill = read_data(dirc,params["fill_index"])

    # corr_thrs = params["corr_thrs"]
    # cm = get_cmtx_feat(df_fill,corr_thrs,False)
    # top_features = nlargest(len(cm),cm,key=cm.get)

    top_features = get_headers(df_fill)
    model_build(df_fill,top_features)
    # get_history(df_fill,top_features)    
