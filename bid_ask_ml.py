#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import xgboost as xgb
import statsmodels.api as sm
import seaborn as sns
import pickle
import sklearn
import mlxtend
import statsmodels.api as sm
import joblib
from pylab import *
from matplotlib import style
from heapq import nlargest
from pandas.plotting import register_matplotlib_converters,scatter_matrix
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestRegressor, \
    ExtraTreesRegressor,VotingClassifier,AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.externals import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from tpot import TPOTRegressor,TPOTClassifier
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, \
    RobustScaler, PowerTransformer, QuantileTransformer, Normalizer
from sklearn.metrics import r2_score
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from dask.distributed import Client
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from mlxtend.regressor import StackingRegressor
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from numpy.random import rand

pd.options.mode.chained_assignment = None 

# [ 'lin', 'svm-lin',svm-poly', 'tree', 'forest', 'xgbr', 'nn', 'comb', 'tpot', 'ada', 'lasso', 'voting']

params = {
    'check_shape': True,
    'test_size': 0.2,
    'regrs': 'xgbr',
    'grid_search': False,
    'loss_eval': False,
    'filename': 'model.pkl',
    'lasso_eps': 0.0001,
    'lasso_nalpha': 1000,
    'lasso_iter': 100000,
    'deg_poly': 2
}

def get_headers(df):
    return(df.columns.values)

def read_data(path,methd):
    methods = ['ffill','bfill','mean','zero']
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
        df.fillna(0,inplace=True)
        return(df)
    elif(methd not in methods):
        df = pd.read_csv(path,sep=',')
        return(df)
    
def view_data(df):
    print(df.head(10))

def get_info(df):  
    df.info()
    df.describe()
    
def format_output(df):
    df.columns = [''] * len(df.columns)
    df = df.to_string(index=False)
    return(df)
   
def check_lin(df):
    cols = [ col for col in df.columns if 'y' not in col ]
    for el in cols:
        fig = plt.figure() 
        plt.scatter(df[str(el)], df['y'], color='red')
        plt.xlabel(str(el), fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.grid(True)
        fig.savefig("figs/y_vs_"+str(el), bbox_inches='tight')
        plt.close()
      
def check_features(df,flag):
    cols = [ col for col in df.columns if str(flag) in col ] + ['y']
    scatter_matrix(df[cols], figsize=(15, 10), diagonal='kde')
    plt.show()

def get_cmtx_feat(df,corr_thrs,plotting):
    corr_matrix = df.corr()
    if(plotting):
        s=sns.heatmap(corr_matrix,cmap='coolwarm') 
        s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
        s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)
        plt.show()
    cmtx = corr_matrix["y"].sort_values(ascending=False)    
    dct = dict(zip(list(cmtx.keys()),cmtx.values))
    dct_select = dict((k, v) for k, v in dct.items() if v >= float(corr_thrs)/100.0 and k != 'y')
    return(dct_select)

def plot_focus(df,top_features):
    df.plot(kind="scatter", x=str(top_features[1]), y="y", alpha=0.5)
    plt.show()

def plot_dist(df,feat):
    arr = np.array(df[str(feat)])
    raw_plt = sns.distplot(arr)
    # sqr_plt = sns.distplot(np.sqrt(arr))
    log_plt = sns.distplot(np.log(arr))
    plt.show()    
    
def check_shape(X_train, X_test, y_train, y_test):
    print('X_train.shape, X_test.shape = ', X_train.shape, X_test.shape)
    print('y_train.shape, y_test.shape = ', y_train.shape, y_test.shape)
   
def ml_train_test(df,top_features):
    cols = top_features + ['y']
    df = df[cols]
   
    X = np.array(df.drop(['y'],axis=1))
    y = np.array(df['y'])  
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=params['test_size'],random_state=47252)
    if(params['check_shape']):
        check_shape(X_train, X_test, y_train, y_test)
    return(X_train, X_test, y_train, y_test)

def ml_train_test_kf(df,top_features):
    cols = top_features + ['y']
    df = df[cols]
   
    X = np.array(df.drop(['y'],axis=1))
    y = np.array(df['y'])  
         
    return(X,y)

def regressors(regrs):
    if(regrs == 'lin'):
        reg = LinearRegression(n_jobs=-1)
    elif(regrs == 'svm-lin'):
        reg = svm.SVR(kernel='linear',gamma='auto')
    elif(regrs == 'svm-poly'):
        reg = svm.SVR(kernel='poly',gamma='auto')
    elif(regrs == 'lasso'):
        reg = make_pipeline(PolynomialFeatures(params['deg_poly'], interaction_only=False), LassoCV(eps=params['lasso_eps'],\
                            n_alphas=params['lasso_nalpha'],max_iter=params['lasso_iter'], normalize=False,cv=5))
    elif(regrs == 'tree'):
        reg = DecisionTreeRegressor(random_state=24361)
    elif(regrs == 'forest'):
        reg = RandomForestRegressor(n_estimators=20,max_depth=2,min_samples_split=4,min_samples_leaf=1,random_state=24361,n_jobs=-1) 
    elif(regrs == 'xgbr'):
        reg=XGBRegressor(learning_rate=0.10, max_depth=2, min_child_weight=1, \
                         n_estimators=100, subsample=0.25)
        # reg = XGBRegressor(learning_rate=0.045, max_depth=2, min_child_weight=1, \
        #                    n_estimators=100, subsample=0.15
        # eta=0.2, gamma=0.9, reg_lambda=0.1, reg_alpha=0.3, n_jobs=-1
    elif(regrs == 'ada'):
        nn = MLPRegressor(hidden_layer_sizes=(32,1), activation='relu', solver='adam', random_state=24361)
        xgbr=XGBRegressor(learning_rate=0.10, max_depth=2, min_child_weight=1, \
                         n_estimators=100, subsample=0.25, random_state=24361)
        # xgbr = XGBRegressor(learning_rate=0.045, max_depth=2, min_child_weight=1, \
        #                     n_estimators=100, subsample=0.15, gamma=0.3, reg_lambda=0.5, reg_alpha=0.4, n_jobs=-1)
        reg = AdaBoostRegressor(base_estimator=xgbr, learning_rate=0.1, loss='square', \
                                n_estimators=100, random_state=24361)
    elif(regrs == 'nn'):
        reg = MLPRegressor(hidden_layer_sizes=(32,1), activation='relu', solver='adam', random_state=24361)
                           # learning_rate='constant', learning_rate_init=0.01, alpha=0.001, power_t=0.5, max_iter=50, \
                           # tol=0.0001, momentum=0.5, nesterovs_momentum=True, validation_fraction=0.1, \
                           # beta_1=0.1, beta_2=0.555, epsilon=1e-08, n_iter_no_change=50, random_state=24361)        
    elif(regrs == 'comb'):
        xgbr = XGBRegressor(learning_rate=0.045, max_depth=2, min_child_weight=1, \
                            n_estimators=100, subsample=0.15, n_jobs=-1)
        xgbr1 = XGBRegressor(learning_rate=0.035, max_depth=3, min_child_weight=1, \
                            n_estimators=50, subsample=0.15, n_jobs=-1)
        # xgbr2 = XGBRegressor(learning_rate=0.025, max_depth=2, min_child_weight=1, \
        #                     n_estimators=50, subsample=0.15, n_jobs=-1)
        frst = RandomForestRegressor(max_depth=2, max_leaf_nodes=2, n_estimators=3, n_jobs=-1)
        dtr = DecisionTreeRegressor(max_depth=2,max_leaf_nodes=2)
        nn = MLPRegressor(hidden_layer_sizes=(32,1), activation='tanh', solver='adam', learning_rate_init=0.15)  
        reg = StackingRegressor(regressors=[xgbr,xgbr1,frst,nn],meta_regressor=frst)
    elif(regrs == 'tpot'):
        reg = TPOTRegressor(generations=10,verbosity=2,scoring='r2',n_jobs=-1,random_state=23)
    elif(regrs == 'voting'):
        frst = RandomForestRegressor(n_estimators=100,random_state=24361,n_jobs=-1)
        dtr = DecisionTreeRegressor(random_state=24361)
        reg = VotingClassifier(estimators=[('frst',frst),('dtr',dtr)],voting='hard')
    return(reg)

def set_grid_search(regrs,reg):
    if(regrs=='forest'):
        random_grid = build_grid_rf()
        prms = grid_search(reg,X_train,y_train,random_grid)
        reg_prms = RandomForestRegressor(n_estimators=prms['n_estimators'],max_features=prms['max_features'], \
                                         max_depth=prms['max_depth'], min_samples_split=prms['min_samples_split'], \
                                         min_samples_leaf=prms['min_samples_leaf'],random_state=24361, n_jobs=-1) 
    elif(regrs=='xgbr'):
        random_grid = build_grid_xgbr()
        prms = grid_search(reg,X_train,y_train,random_grid)
        reg_prms = XGBRegressor(learning_rate=prms['learning_rate'], max_depth=prms['max_depth'], \
                                min_child_weight=prms['min_child_weight'], n_estimators=prms['n_estimators'],\
                                subsample=prms['subsample'], random_state=24361, n_jobs=-1)
        # gamma = prms['gamma'], \
        #                         reg_lambda = prms['reg_lambda'], reg_alpha = prms['reg_alpha'], \
    elif(regrs=='nn'):
        random_grid = build_grid_nn()
        prms = grid_search(reg,X_train,y_train,random_grid)
        reg_prms = MLPRegressor(hidden_layer_sizes=prms['hidden_layer_sizes'],activation=prms['activation'],solver=prms['solver'],\
                                alpha=prms['alpha'],learning_rate_init=prms['learning_rate_init'],learning_rate=prms['learning_rate'],\
                                max_iter=prms['max_iter'],tol=prms['tol'],momentum=prms['momentum'],beta_1=prms['beta_1'],\
                                beta_2=prms['beta_2'],n_iter_no_change=prms['n_iter_no_change'],random_state=24361)
    elif(regrs=='ada'):
        random_grid = build_grid_ada()
        prms = grid_search(reg,X_train,y_train,random_grid)
        xgbr = XGBRegressor(learning_rate=0.045, max_depth=2, min_child_weight=1, \
                           n_estimators=105, subsample=0.20, gamma=0.3, reg_lambda=0.1, reg_alpha=0.3, n_jobs=-1)
        reg_prms = AdaBoostRegressor(base_estimator=None, learning_rate=prms['learning_rate'],n_estimators=prms['n_estimators'],\
                                     random_state=24361)
    return(reg_prms)

def loss_eval(X_train,X_test,y_train,y_test,reg):
    eval_set = [(X_train, y_train), (X_test, y_test)]
    reg.fit(X_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)

    ################################################################################################
    # eval_set = [(X_test, y_test)]
    # reg.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
    # reg.fit(X_train, y_train, eval_metric="error", eval_set=eval_set, verbose=True)
    ################################################################################################

    results = reg.evals_result()
    epochs = len(results['validation_0']['error'])
    x_axis = range(0, epochs)

    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='train')
    ax.plot(x_axis, results['validation_1']['logloss'], label='test')
    ax.legend()
    plt.ylabel('log-loss')
    plt.title('XGBoost log-loss')
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['error'], label='train')
    ax.plot(x_axis, results['validation_1']['error'], label='test')
    ax.legend()
    plt.ylabel('error')
    plt.title('XGBoost error')
    plt.show()
    
    return(reg)
    
def model(X_train,X_test,y_train,y_test):
    # print("starting model..")
    # start_time = time.time()

    if(params['grid_search']):
        reg_init = regressors(params['regrs'])
        reg = set_grid_search(params['regrs'],reg_init)
    else:
        reg = regressors(params['regrs'])
       
    if(params['loss_eval']):
        reg = loss_eval(X_train,X_test,y_train,y_test,reg)
    else:
        reg.fit(X_train,y_train)
        
    predictions = reg.predict(X_test)
    joblib.dump(reg,params['filename'])

    accuracy = r2_score(y_test,predictions)

    # print("ending model:", time.time() - start_time)
    print(accuracy)

def model_kf(X,y):
    reg = regressors(params['regrs'])       
    scores = cross_val_score(reg, X, y, scoring='r2', cv=10)
    print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))
   
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
    
def plot_ml(test,pred):
    plt.plot(test,'r*-')
    plt.plot(pred,'bo-')
    plt.xlabel('y - true values')
    plt.ylabel('y - predictions')
    plt.show()      

def display_scores(scores):
    print('###########################################################')
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    print('###########################################################')

def display_output(acc,pred,act_vals):
    print('###########################################################')
    print('accuracy    = ', acc)
    print('y predictions = ', list(map(lambda x: '{0:.2f}'.format(x),list(pred))))
    print('actual vals = ', list(act_vals))
    print('###########################################################')


def model_tpot(X_train,X_test,y_train,y_test):
    reg = regressors(params['regrs'])       
    reg.fit(X_train,y_train)        

    predictions = reg.predict(X_test)       
    joblib.dump(reg,params['filename'])

    accuracy = r2_score(y_test,predictions)
    print(accuracy)
    
def build_grid_rf():
    # Number of trees in random forest
    n_estimators = [20, 40, 80, 160]
    # [int(x) for x in np.linspace(start = 20, stop = 120, num = 20)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [1, 3, 9]
    # [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    # bootstrap = [True, False]

    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf
    }
    return(grid)

def build_grid_xgbr():
    learning_rate = [float(x) for x in np.linspace(start = 0.045, stop = 0.050, num = 21)]
    n_estimators = [int(x) for x in np.linspace(start = 95, stop = 105, num = 10)]
    max_depth = [ 2 ] # [int(x) for x in np.linspace(1, 10, num = 5)]
    min_child_weight = [ 1 ] # [int(x) for x in np.linspace(1, 20, num = 5)]
    subsample = [0.15] # [float(x) for x in np.linspace(start = 0.05, stop = 0.5, num = 10)]
    # gamma = [0.01, 0.03, 0.1, 0.3, 1.0]
    # reg_lambda = [0.01, 0.03, 0.1, 0.3, 1.0]
    # reg_alpha = [0.01, 0.03, 0.1, 0.3, 1.0]
    
    grid = {'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            # 'gamma': gamma,
            # 'reg_lambda': reg_lambda,
            # 'reg_alpha': reg_alpha
    }
    
    return(grid)

def build_grid_ada():
    learning_rate = [float(x) for x in np.linspace(start = 0.1, stop = 0.9, num = 9)]
    n_estimators = [int(x) for x in np.linspace(start = 20, stop = 150, num = 14)]
        
    grid = {'learning_rate': learning_rate,
            'n_estimators': n_estimators
    }
    return(grid)
    
def build_grid_nn():
    hidden_layer_sizes = [ (128,128), (64,64), (16,16), (8,8), (4,4) ]
    activation = ['tanh']
    solver = ['adam']
    alpha = [float(x) for x in np.linspace(start = 0.001, stop = 0.01, num = 10)]
    learning_rate_init = [float(x) for x in np.linspace(start = 0.001, stop = 0.01, num = 9)]
    learning_rate = ['constant', 'adaptive'] 
    max_iter = [int(x) for x in np.linspace(50, 300, num = 6)]
    tol = [0.0001]
    momentum = [0.1,0.5,0.9]
    beta_1=[0.1,0.3,0.5,0.9]
    beta_2=[0.1,0.5,0.9]
    n_iter_no_change=[5,10,20]

    grid = {'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'learning_rate_init': learning_rate_init,
            'learning_rate': learning_rate,           
            'max_iter': max_iter,
            'tol': tol,
            'momentum': momentum,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'n_iter_no_change': n_iter_no_change
    }
    return(grid)

def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

def grid_search(reg,X_train,y_train,random_grid):
    # print('#############################################')
    # print('parameters in use before grid search')
    # print('#############################################')

    # pprint(reg.get_params())
    # prms = reg.get_params()
    
    reg_random = RandomizedSearchCV(estimator=reg,param_distributions=random_grid,n_iter=10,cv=10, \
                                    verbose=0,random_state=42,n_jobs=-1)
    print("Randomized search..")
    search_time_start = time.time()
    reg_random.fit(X_train,y_train)
    print("Randomized search time:", time.time() - search_time_start)

    prms = reg_random.best_params_

    print('#############################################')
    print('best parameters after grid search')
    print('#############################################')
    pprint(prms)

    return(prms)

def get_skew(df):
    skew_feats=df.skew().sort_values(ascending=False)
    return(skew_feats)

def ml_train_test(df,top_features):
    cols = top_features + ['y']
    df = df[cols]
    
    X = np.array(df.drop(['y'],axis=1))
    y = np.array(df['y'])  

    print(df['y'].shape)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=params['test_size'],random_state=47252)
    if(params['check_shape']):
        check_shape(X_train, X_test, y_train, y_test)
    return(X_train, X_test, y_train, y_test)

def ols_reg(df,top_features):
    cols = top_features + ['y']
    df = df[cols]
   
    X = np.array(df.drop(['y'],axis=1))
    X = sm.add_constant(X)
    y = np.array(df['y'])  

    model = sm.OLS(y,X)
    results = model.fit()

    predictions = results.predict(X)       
    joblib.dump(model,params['filename'])

    accuracy = r2_score(y,predictions)

    print(results.summary())
    print(accuracy)

def count_pos(df):
    cols = df.columns.values
    dct_pos = {}
    for el in cols:
        dct_pos[str(el)] = np.sum(df[str(el)] > 0)
    return(dct_pos)

def decorrelate(X):
    # Calculate the eigenvalues and eigenvectors of the covariance matrix
    eigVals, eigVecs = np.linalg.eig(X)
    # Apply the eigenvectors to X
    decorrelated = X.dot(eigVecs)
    return(decorrelated)

if __name__ == '__main__':
    regrs = params['regrs']
    dirc = os.path.join(os.getcwd(),'data_sample_1e3.csv')

    df_raw = read_data(dirc,'zero')   
    print('df_raw = ', df_raw.shape)
    # cmtx = corr_matrix["y"].sort_values(ascending=False)    
    # dct = dict(zip(list(cmtx.keys()),cmtx.values))
    # dct_select = dict((k, v) for k, v in dct.items() if v >= float(corr_thrs)/100.0 and k != 'y')
    
    # minval, maxval = 0, 5734
    # rst = int(minval + (rand(1) * (maxval - minval)))
    # df_raw_new = df_raw.sample(frac=1.0,random_state=rst).reset_index(drop=True)

    # sample = True
    # if(sample):
    #     df_run = df_raw_new
    # else:
    #     df_run = df_raw
        
    # (0) for adding more samples
    # df_raw_sample = df_raw.sample(frac=1.0).reset_index(drop=True)
    # df_new = pd.concat([df_raw, df_raw_sample], ignore_index=True)
    
    # (1) plot features scatterplot
    # flags, index = [ 'bidRate0', 'bidSize0', 'askRate0', 'askSize0' ], 1
    # check_features(df_raw,flags[index])

    # (2) correlation matrix
    # corr_thrs = 1 
    # cm = get_cmtx_feat(df_raw,corr_thrs,True)
    # get top features after threshold selection
    top_features = list(get_headers(df_raw))[:-1] # nlargest(len(cm),cm,key=cm.get)
    # print(top_features)
    
    # (3) basic ml regressors
    X_train, X_test, y_train, y_test = ml_train_test(df_raw,top_features)
    model(X_train,X_test,y_train,y_test)
   
    # (3kf) basic ml regressors
    # X, y = ml_train_test_kf(df_raw,top_features)
    # model_kf(X,y)   
    
    # display_output(accuracy,predictions,y_test)
    # plot_ml(y_test,predictions)
   
    # (4) test auto ML (using Tpot)
    # model_tpot(X_train,X_test,y_train,y_test)

    # (5) multivariate linear regression
    # ols_reg(df_raw,top_features)
    

