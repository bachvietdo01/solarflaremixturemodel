#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:45:34 2021

@author: vietdo
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from SolarFlareMM1 import SolarFlareMM1
from SolarFlareMM1Sim import SolarFlareMM1Sim

# simulate data
N = 1000
D = 2
K = 2

mu = np.zeros((K, D))
Sigma = np.zeros((K, D, D))
sigma2 = 1.0


pi = np.array([0.2, 0.8])
mu[0,] = [-5, 0]
mu[1,] = [5, 0]
Sigma[0,] = 4 * np.identity(D)
Sigma[1,] = np.identity(D)

mm1_sim = SolarFlareMM1Sim(N, D, K, sigma2, pi, mu, Sigma)
mm1_sim.generate()

# initalize Solar Flare MM1
X = mm1_sim.X[:800,]
y = mm1_sim.y[:800,]
X_test = mm1_sim.X[800:,]
y_test = mm1_sim.y[800:,]


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
    

# Estimator
    
beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
y_pred = X_test.dot(beta_hat)
rmse = np.sqrt(np.sum(np.square(y_pred - y_test)) / y_test.shape[0])
print("Linear Regerssion RMSE is {}".format(rmse))


# Gibbs Sampler
K = 2

chain1 = run_mm1_gibbs(1000, X, y, K, X_test, y_test)
chain2 = run_mm1_gibbs(1000, X, y, K, X_test, y_test)
chain3 = run_mm1_gibbs(1000, X, y, K, X_test, y_test)
chain4 = run_mm1_gibbs(1000, X, y, K, X_test, y_test)

    
# traceplot
burin_iter = 100


# sigma2 trace plot
fig = plt.figure()
ax = plt.axes()
ax.plot(range(burin_iter, chain1['sigma2'].shape[0]), chain1['sigma2'][burin_iter:].tolist())
ax.plot(range(burin_iter, chain2['sigma2'].shape[0]), chain2['sigma2'][burin_iter:].tolist())
ax.plot(range(burin_iter, chain3['sigma2'].shape[0]), chain3['sigma2'][burin_iter:].tolist())
ax.plot(range(burin_iter, chain4['sigma2'].shape[0]), chain4['sigma2'][burin_iter:].tolist())
plt.show()


# pi trace plot
for i in range(K):    
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(range(burin_iter, chain1['pi'].shape[0]), chain1['pi'][burin_iter:, i].tolist())
    ax.plot(range(burin_iter, chain2['pi'].shape[0]), chain2['pi'][burin_iter:, i].tolist())
    ax.plot(range(burin_iter, chain3['pi'].shape[0]), chain3['pi'][burin_iter:, i].tolist())
    ax.plot(range(burin_iter, chain4['pi'].shape[0]), chain4['pi'][burin_iter:, i].tolist())
    plt.show()

# mu trace plot
for i in range(K):    
    for j in range(X.shape[1]):
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(range(burin_iter, chain1['mu'].shape[0]), chain1['mu'][burin_iter:, i,j].tolist())
        ax.plot(range(burin_iter, chain2['mu'].shape[0]), chain2['mu'][burin_iter:, i,j].tolist())
        ax.plot(range(burin_iter, chain3['mu'].shape[0]), chain3['mu'][burin_iter:, i,j].tolist())
        ax.plot(range(burin_iter, chain4['mu'].shape[0]), chain4['mu'][burin_iter:, i,j].tolist())
        plt.show()
        
# Sigma trace plot
for i in range(K):    
    for j in range(X.shape[1]):
        for k in range(X.shape[1]):
            fig = plt.figure()
            ax = plt.axes()
            ax.plot(range(burin_iter, chain1['Sigma'].shape[0]), chain1['Sigma'][burin_iter:, i,j, k].tolist())
            ax.plot(range(burin_iter, chain2['Sigma'].shape[0]), chain2['Sigma'][burin_iter:, i,j, k].tolist())
            ax.plot(range(burin_iter, chain3['Sigma'].shape[0]), chain3['Sigma'][burin_iter:, i,j, k].tolist())
            ax.plot(range(burin_iter, chain4['Sigma'].shape[0]), chain4['Sigma'][burin_iter:, i,j, k].tolist())
            plt.show()