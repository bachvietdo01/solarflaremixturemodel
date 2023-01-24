#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 19:28:56 2021

@author: vietdo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:18:20 2021

@author: vietdo
"""
import os as os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.random import randint
from SolarFlareMM2REM import SolarFlareMM2REM
from SolarFlareMM2AdEM import SolarFlareMM2AdEM
from trainutilities import load_raw_train_and_test_set_split_by_AR_coinflip, create_train_and_val_set_split_by_AR_coinflip, \
                            region_and_long_lat, create_W_by_fit_st, LinearRegression, get_model_predict_values

def run_mm2R_em(niters, R, X, y, K, W,  beta0):    
    print(f"Executing run_mm2R_em with {K:,} on pid {os.getpid()}")
    D = X.shape[1]
    
    beta_init = np.zeros((K, D))
    for k in range(K):
        beta_init[k,] = beta0 + uniform.rvs(-.1, .1,  D)
    
    mm2R = SolarFlareMM2REM(R, X, y, K, W = W, beta0 = beta_init)
    
    for i in range(niters):    
        mm2R.EM_iter()
    
    return { 'mm2R': mm2R}

def run_mm2Ad_em(niters, R, X, y, K, W, mm2r):    
    print(f"Executing run_mm2Ad_em with {K:,} on pid {os.getpid()}")
    beta0 =  mm2r.beta
    pir0 = mm2r.tau
    
    D = X.shape[1]
    
    beta_init = np.zeros((K, D))
    for k in range(K):
        beta_init[k,] = beta0[k,] + uniform.rvs(-.1, .1, D)
        
    mm2Ad = SolarFlareMM2AdEM(R, X, y, K, W = W, beta0 = beta_init, pir0 = pir0)

    for i in range(niters):    
        mm2Ad.EM_iter()
        
    
    return {'mm2Ad': mm2Ad}


HOURS_BEHIND = 6

data_path  = '../fulldata/arflares' + str(HOURS_BEHIND) +  'h.csv'

X, y, R_test, X_test, y_test, pca_xtest, pca_ts = load_raw_train_and_test_set_split_by_AR_coinflip(data_path, 
                                      ubr = None, ts = 0.2, seed_no = randint(low=1, high=100))
R_test, loc_test = region_and_long_lat(R_test)


def prepare_run_data(X, y):
    R, X_train, y_train, pca_xtrain, pca_tr, \
    R_val, X_val, y_val, pca_xval, pca_val  = create_train_and_val_set_split_by_AR_coinflip(X, y, 
                                                                                   vs = 0.2,
                                                            seed_no = randint(low=1, high=100))

    R, loc = region_and_long_lat(R)
    R_val, loc_val = region_and_long_lat(R_val)
    
    return R, loc, X_train, y_train, pca_xtrain, pca_tr, \
           R_val, loc_val, X_val, y_val, pca_xval, pca_val

# create validate set
R, loc, X_train, y_train, pca_xtrain, pca_tr, \
        R_val, loc_val, X_val, y_val, pca_xval, pca_val = prepare_run_data(X, y)


def save_files(R, X, y, path):
     pd.DataFrame(R).to_csv(path + '_R.csv', index = False)
     pd.DataFrame(X).to_csv(path + '_X.csv', index = False)
     pd.DataFrame(y).to_csv(path + '_y.csv', index = False)

from scipy.stats import uniform

      
# Run mooldes
offset = 0
Nrep = 20
MAX_K = 7
RITERS = 250
HITERS = 500

train_size, val_size, test_size = X_train.shape[0], X_val.shape[0], X_test.shape[0]


train_metrics2r = np.zeros((MAX_K, train_size, 2*  Nrep))
test_metrics2r = np.zeros((MAX_K, test_size, 2 * Nrep))
val_metrics2r = np.zeros((MAX_K, val_size, 2 * Nrep))

train_metrics2h = np.zeros((MAX_K, train_size, 2 * Nrep))
test_metrics2h = np.zeros((MAX_K, test_size, 2 * Nrep))
val_metrics2h = np.zeros((MAX_K, val_size, 2 * Nrep))


prefix_path = '../dump/' + str(HOURS_BEHIND) + 'h/'
save_files(R_test, X_test, y_test, prefix_path + '_test')


for rep in range(Nrep):
    # create validate set
    R, loc, X_train, y_train, pca_xtrain, pca_tr, \
            R_val, loc_val, X_val, y_val, pca_xval, pca_val = prepare_run_data(X, y)
    
    
    W = create_W_by_fit_st(1, y_train)
    wlr = LinearRegression(X_train, y_train, W = W)
    
    train_metrics2r[0,:,Nrep + rep] = y_train.copy()
    test_metrics2r[0,:,Nrep + rep] = y_test.copy()
    val_metrics2r[0, : ,Nrep + rep] = y_val.copy()

    train_metrics2h[0,:,Nrep + rep] = y_train.copy()
    test_metrics2h[0,:,Nrep + rep] =  y_test.copy()
    val_metrics2h[0,:, Nrep + rep] = y_val.copy()
    
    train_metrics2r[0,:, rep] = get_model_predict_values(wlr, R, X_train)
    test_metrics2r[0,:, rep] = get_model_predict_values(wlr, R_test, X_test)
    val_metrics2r[0,:, rep] = get_model_predict_values(wlr, R_val, X_val)
        
    train_metrics2h[0,:, rep] = get_model_predict_values(wlr, R, X_train)
    test_metrics2h[0,:, rep] = get_model_predict_values(wlr, R_test, X_test)
    val_metrics2h[0,:, rep] = get_model_predict_values(wlr, R_val, X_val)
        
    
    file_path = prefix_path + 'r' + str(rep)
    save_files(R, X_train, y_train, file_path + '_train')
    save_files(R_val, X_val, y_val, file_path + '_val')
    
    
    # run model 2R in parallel    
    results_2r = Parallel(n_jobs=MAX_K - 1, verbose=1) \
                         (delayed(run_mm2R_em)(RITERS, R, X_train, y_train,K, W, wlr.beta_hat) \
                                                   for K in range(2, MAX_K + 1))

    for K in range(2, MAX_K + 1):
        # run model 2R
        obj_name = prefix_path + 'mm2RK' + str(K) + 'r' + str(rep) + '.pkl'
        print("Processing " + obj_name + " ...")
                
        em_run2R = results_2r[K - 2]
    
        #with open(obj_name, 'wb') as f:
            #pickle.dump(em_run2R, f, pickle.HIGHEST_PROTOCOL)
            
            
        train_metrics2r[K-1,:, Nrep + rep] = y_train.copy()
        test_metrics2r[K-1,:, Nrep + rep] = y_test.copy()
        val_metrics2r[K-1, : , Nrep + rep] = y_val.copy()
            
        train_metrics2r[K-1, :, rep] = get_model_predict_values(em_run2R['mm2R'], R, X_train)
        test_metrics2r[K-1, :, rep] = get_model_predict_values(em_run2R['mm2R'], R_test, X_test)
        val_metrics2r[K-1, :, rep] = get_model_predict_values(em_run2R['mm2R'], R_val, X_val)
    
    # run model 2H in parallel        
    results_2h = Parallel(n_jobs=MAX_K - 1, verbose=1) \
                         (delayed(run_mm2Ad_em)(HITERS, R, X_train, y_train, K, W, \
                                                      results_2r[K - 2]['mm2R']) \
                                                for K in range(2, MAX_K + 1))

    for K in range(2, MAX_K + 1):
        # run model 2R
        obj_name = prefix_path + 'mm2HK' + str(K) + 'r' + str(rep) + '.pkl'
        print("Processing " + obj_name + " ...")
                
        em_run2H = results_2h[K - 2]
    
        #with open(obj_name, 'wb') as f:
            #pickle.dump(em_run2H, f, pickle.HIGHEST_PROTOCOL)
        
        train_metrics2h[K-1,:,Nrep + rep] = y_train.copy()
        test_metrics2h[K-1,:,Nrep + rep] =  y_test.copy()
        val_metrics2h[K-1,:,Nrep + rep] = y_val.copy()
        
    
        train_metrics2h[K-1,:,rep] = get_model_predict_values(em_run2H['mm2Ad'], R, X_train)
        test_metrics2h[K-1,:,rep] = get_model_predict_values(em_run2H['mm2Ad'], R_test, X_test)
        val_metrics2h[K-1,:,rep] = get_model_predict_values(em_run2H['mm2Ad'], R_val, X_val)
    

for K in range(MAX_K):
    pd.DataFrame(train_metrics2r[K,]).to_csv('../results/2R/' + str(HOURS_BEHIND) +  'htrain_metricsK' + str(K) + '.csv', header =False, index=False)
    pd.DataFrame(val_metrics2r[K,]).to_csv('../results/2R/' + str(HOURS_BEHIND) +  'hval_metricsK' + str(K) + '.csv', header =False, index=False)
    pd.DataFrame(test_metrics2r[K,]).to_csv('../results/2R/' + str(HOURS_BEHIND) + 'htest_metricsK' + str(K) + '.csv', header =False, index=False)
    
        
for K in range(MAX_K):
    pd.DataFrame(train_metrics2h[K,]).to_csv('../results/2H/' + str(HOURS_BEHIND) +  'htrain_metricsK' + str(K) + '.csv', header =False, index=False)
    pd.DataFrame(val_metrics2h[K,]).to_csv('../results/2H/' + str(HOURS_BEHIND) +  'hval_metricsK' + str(K) + '.csv', header =False, index=False)
    pd.DataFrame(test_metrics2h[K,]).to_csv('../results/2H/' + str(HOURS_BEHIND) + 'htest_metricsK' + str(K) + '.csv', header =False, index=False)
    















    



