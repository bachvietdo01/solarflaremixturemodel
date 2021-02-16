#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 12:27:45 2021

@author: vietdo
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from SolarFlareMM1 import SolarFlareMM1
from SolarFlareMM1Sim import SolarFlareMM1Sim

X_train = pd.read_csv('Xtrainpca3.csv').to_numpy()
X_test = pd.read_csv('Xtestpca3.csv').to_numpy()
y_train = pd.read_csv('ytrain.csv').to_numpy()
y_test = pd.read_csv('ytest.csv').to_numpy()

def run_mm2_em(niters, X, y, K, X_test = None, y_test = None, mm = None,
               debug_r = None, debug_beta = None, debug_sigma2 = None, debug_pi = None,
               pi_0 = None):
    N = X.shape[0]
    D = X.shape[1]
        
    # tracking model parameters
    pi_ts = np.zeros((niters, K))
    beta_ts = np.zeros((niters, K, D))
    sigma2_ts = np.zeros((niters,K))
    
    r_ts = np.zeros((niters, N, K))
    rmse_ts = np.zeros(niters)
    rmse_alt_ts = np.zeros(niters)
    ll_ts = np.zeros(niters)
    aic_ts = np.zeros(niters)
    bic_ts = np.zeros(niters)
    
    if mm is None:
        mm2 = SolarFlareMM2EM(X, y, K, pi_0 = pi_0, debug_sigma2 = debug_sigma2,
                              debug_pi = debug_pi, debug_r = debug_r, debug_beta = debug_beta)
    else:
        mm2 = mm

    for i in range(niters):    
        mm2.EM_iter()
        mm2.compute_model_selection_criteria()
        rmse_ts[i] = mm2.compute_rmse(X_test, y_test)
        rmse_alt_ts[i] = mm2.compute_rmse_alt(X_test, y_test)
        
        pi_ts[i, ] = mm2.pi
        beta_ts[i,] = mm2.beta
        sigma2_ts[i,] = mm2.sigma2
        ll_ts[i] =mm2.ll
        aic_ts[i] = mm2.aic
        bic_ts[i] = mm2.bic
    
        if i % 1 == 0:
            print("Iteration {}.".format(i))
            print(beta_ts[i,])
            print(pi_ts[i, ])
            print(sigma2_ts[i])
            print("RMSE is {}.".format(rmse_ts[i]))
            print("Alt RMSE is {}.".format(rmse_alt_ts[i]))
            print("Log likelihood is {}.".format(ll_ts[i]))
            print("AIC is {}.".format(aic_ts[i]))
            print("BIC is {}.".format(bic_ts[i]))
    
    return {'pi': pi_ts, 'beta': beta_ts, 'r': r_ts, 'sigma2':sigma2_ts, 'mm2': mm2, 'rmse': rmse_ts}
    

# Linear Regerssion MLE
beta_hat = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
y_pred = X_test.dot(beta_hat)
rmse = np.sqrt(np.sum(np.square(y_pred - y_test)) / y_test.shape[0])
print("Linear Regerssion RMSE is {}".format(rmse))


K = 15

pi0 = dist.dirichlet.rvs(np.full(K, 1))[0]

em_run = run_mm2_em(100, X_train, y_train, K, X_test, y_test, pi_0 = pi0)