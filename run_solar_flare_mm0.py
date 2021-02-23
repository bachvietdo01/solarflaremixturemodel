#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:18:20 2021

@author: vietdo
"""

import matplotlib.pyplot as plt
import scipy.stats as dist
import pandas as pd
import numpy as np
from SolarFlareMM0EM import SolarFlareMM0EM

X_train = pd.read_csv('data/Xtrain.csv').to_numpy()
X_test = pd.read_csv('data/Xtest.csv').to_numpy()
y_train = pd.read_csv('data/ytrain.csv').to_numpy()
y_test = pd.read_csv('data/ytest.csv').to_numpy()

def run_mm0_em(niters, X, y, K, X_test = None, y_test = None, mm = None,
               debug_r = None, debug_beta = None, debug_sigma2 = None, debug_pi = None,
               debug_mu = None, debug_Sigma = None, pi0 = None):
    D = X.shape[1]
        
    # tracking model parameters
    pi_ts = np.zeros((niters, K))
    beta_ts = np.zeros((niters, K, D))
    sigma2_ts = np.zeros((niters,K))
    mu_ts = np.zeros((niters, K, D))
    Sigma_ts = np.zeros((niters, K, D, D))
    rmse_ts = np.zeros(niters)
    logll_ts = np.zeros(niters)
    aic_ts = np.zeros(niters)
    bic_ts = np.zeros(niters)
    ecll_ts = np.zeros(niters)
    
    
    if mm is None:
        mm0 = SolarFlareMM0EM(X, y, K, debug_sigma2 = debug_sigma2,
                              debug_pi = debug_pi, debug_r = debug_r, debug_beta = debug_beta,
                              debug_mu = debug_mu, debug_Sigma = debug_Sigma, pi0 = pi0)
    else:
        mm0 = mm

    for i in range(niters):    
        mm0.EM_iter()
        mm0.compute_selection_cretia()
        
        rmse_ts[i] = mm0.compute_rmse(X_test, y_test)
        pi_ts[i, ] = mm0.pi
        beta_ts[i,] = mm0.beta
        sigma2_ts[i,] = mm0.sigma2
        mu_ts[i, ] = mm0.mu
        Sigma_ts[i,] = mm0.Sigma
        logll_ts[i] = mm0.logll
        aic_ts[i] = mm0.aic
        bic_ts[i] = mm0.bic
        ecll_ts[i] = mm0.ecll

        if i % 5 == 0:
            print("Iteration {}.".format(i))
            print(beta_ts[i,])
            print(sigma2_ts[i])
            print(pi_ts[i, ])
            print(mu_ts[i])
            print(Sigma_ts[i])
            print("rmse is {}".format(rmse_ts[i]))
            print("Expected Complete likehood is {}".format(ecll_ts[i]))
            print("Log likehood is {}".format(logll_ts[i]))
            print("AIC is {}".format(aic_ts[i]))
            print("BIC is {}".format(bic_ts[i]))
        
    
    return {'pi': pi_ts, 'beta': beta_ts, 'sigma2':sigma2_ts, 'mm0': mm0, 'mu': mu_ts,
            'Sigma': Sigma_ts, "log_ll": logll_ts, "aic": aic_ts, "bic": bic_ts,
            'ecll': ecll_ts}
    
# Linear Regerssion MLE
beta_hat = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
y_pred = X_test.dot(beta_hat)
rmse = np.sqrt(np.sum(np.square(y_pred - y_test)) / y_test.shape[0])
print("Linear Regerssion RMSE is {}".format(rmse))


K = 5

pi0 = dist.dirichlet.rvs(np.full(K, 1))[0]
em_run = run_mm0_em(50, X_train, y_train, K, X_test, y_test, pi0 = pi0)

em_run['pi'][-1]

em_run['log_ll'][-1]
em_run['aic'][-1]
em_run['bic'][-1]

# pi_hat
pi_hat = em_run['pi'][-1]
cl_nums = range(len(beta_hat[0]))
plt.bar(cl_nums, beta_hat[0], align='center', alpha=0.5)
plt.xticks(cl_nums, cl_nums)

# visualize beta cofficients
beta_hat = em_run['beta'][-1]


feat_nums = range(len(beta_hat[0]))
plt.bar(feat_nums, beta_hat[0], align='center', alpha=0.5)
plt.xticks(feat_nums, feat_nums)
plt.bar(feat_nums, beta_hat[1], align='center', alpha=0.5)
plt.xticks(feat_nums, feat_nums)
plt.bar(feat_nums, beta_hat[2], align='center', alpha=0.5)
plt.xticks(feat_nums, feat_nums)
plt.bar(feat_nums, beta_hat[3], align='center', alpha=0.5)
plt.xticks(feat_nums, feat_nums)
plt.bar(feat_nums, beta_hat[4], align='center', alpha=0.5)
plt.xticks(feat_nums, feat_nums)
