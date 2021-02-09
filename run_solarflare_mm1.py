#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:23:43 2021

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


def run_mm1_gibbs(niters, X, y, K, X_test = None, y_test = None, mm = None, ):
    N = X.shape[0]
    D = X.shape[1]
        
    # tracking model parameters
    pi_ts = np.zeros((niters, K))
    mu_ts = np.zeros((niters, K, D))
    Sigma_ts = np.zeros((niters, K, D, D))
    beta_ts = np.zeros((niters, N, D))
    z_ts = np.zeros((niters, N))
    sigma2_ts = np.zeros(niters)
    rmse_ts = np.zeros(niters)

    alpha = [1.0] * K
    mu0 = np.zeros(D)
    nu0 = D + 1.0
    kappa0 = 1.0
    Lambda0 =  np.identity(D)
    
    if mm is None:
        mm1 = SolarFlareMM1(X, y, K, alpha, mu0, kappa0, Lambda0, nu0)
    else:
        mm1 = mm

    for i in range(niters):    
        mm1.gibbs_iter()
        
        if X_test is not None and y_test is not None:
            rmse_ts[i] = mm1.compute_rmse(X_test, y_test)
        
        pi_ts[i, ] = mm1.pi
        mu_ts[i,] = mm1.mu
        Sigma_ts[i,] = mm1.Sigma
        beta_ts[i, ] = mm1.beta
        z_ts[i, ] = mm1.z
        sigma2_ts[i] = mm1.sigma2
    
        if i % 25 == 0:
            print("Iteration {}.".format(i))
            print(mu_ts[i,])
            print(Sigma_ts[i,])
            print(pi_ts[i, ])
            print(sigma2_ts[i])
            print(rmse_ts[i])
    
    return {'pi': pi_ts, 'mu' : mu_ts, 'Sigma' : Sigma_ts, 'beta': beta_ts,
            'z': z_ts, 'sigma2':sigma2_ts, 'mm1': mm1, 'rmse': rmse_ts}
    

K = 3
chain1 = run_mm1_gibbs(10000, X_train, y_train, K, X_test = X_test, y_test = y_test, mm = chain1['mm1'])
chain2 = run_mm1_gibbs(10000, X_train, y_train, K, X_test = X_test, y_test = y_test, mm = chain2['mm1'])

# save chains as pickle objects
with open('dump/chain1mm1k3.pickle', 'wb') as handle:
    pickle.dump(chain1, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('dump/chain2mm1k3.pickle', 'wb') as handle:
    pickle.dump(chain2, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load pickle chains
with open('dump/chain1mm1k3.pickle', 'rb') as f:
    chain1 = pickle.load(f)
    
with open('dump/chain2mm1k3.pickle', 'rb') as f:
    chain2 = pickle.load(f)

# traceplot
burin_iter = 1


# sigma2 trace plot
fig = plt.figure()
ax = plt.axes()
ax.plot(range(burin_iter, chain1['sigma2'].shape[0]), chain1['sigma2'][burin_iter:].tolist())
ax.plot(range(burin_iter, chain1['sigma2'].shape[0]), chain2['sigma2'][burin_iter:].tolist())
plt.show()


# pi trace plot
for i in range(K):    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(range(burin_iter, chain1['pi'].shape[0]), chain1['pi'][burin_iter:, i].tolist())
    ax.plot(range(burin_iter, chain1['pi'].shape[0]), chain2['pi'][burin_iter:, i].tolist())
    plt.show()

# mu trace plot
for i in range(K):    
    for j in range(X_train.shape[1]):
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(range(burin_iter, chain1['mu'].shape[0]), chain1['mu'][burin_iter:, i,j].tolist())
        ax.plot(range(burin_iter, chain1['mu'].shape[0]), chain2['mu'][burin_iter:, i,j].tolist())
        plt.show()
        
# Sigma trace plot
for i in range(K):    
    for j in range(X_train.shape[1]):
        for k in range(X_train.shape[1]):
            fig = plt.figure()
            ax = plt.axes()
            ax.plot(range(burin_iter, chain1['Sigma'].shape[0]), chain1['Sigma'][burin_iter:, i,j, k].tolist())
            ax.plot(range(burin_iter, chain1['Sigma'].shape[0]), chain2['Sigma'][burin_iter:, i,j, k].tolist())
            plt.show()