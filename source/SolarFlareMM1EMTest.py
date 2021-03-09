#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:42:40 2021

@author: vietdo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:45:34 2021

@author: vietdo
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
from SolarFlareMM1EM import SolarFlareMM1EM
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

debug_r = np.zeros((N, D))

for n in range(N):
    if mm1_sim.z[n] == 1:
        debug_r[n, 1] = 1
    else:
        debug_r[n, 0] = 1
        
    

def run_mm1_em(niters, X, y, K, X_test = None, y_test = None, mm = None,
               debug_r = None, debug_beta = None, debug_mu = None, debug_Sigma = None,
               debug_pi = None):
    N = X.shape[0]
    D = X.shape[1]
        
    # tracking model parameters
    pi_ts = np.zeros((niters, K))
    mu_ts = np.zeros((niters, K, D))
    Sigma_ts = np.zeros((niters, K, D, D))
    sigma2_ts = np.zeros(niters)
    
    beta_ts = np.zeros((niters, N, D))
    r_ts = np.zeros((niters, N, K))
    rmse_ts = np.zeros((niters, N))
    
    if mm is None:
        mm1 = SolarFlareMM1EM(X, y, K, debug_mu = debug_mu, debug_Sigma = debug_Sigma,
                              debug_pi = debug_pi, debug_r = debug_r, debug_beta = debug_beta,
                              mu_0 = debug_mu, Sigma_0 = debug_Sigma, pi_0 = debug_pi)
    else:
        mm1 = mm

    for i in range(niters):    
        mm1.EM_iter()
        
        
        pi_ts[i, ] = mm1.pi
        mu_ts[i,] = mm1.mu
        Sigma_ts[i,] = mm1.Sigma
        sigma2_ts[i] = mm1.sigma2
    
        if i % 1 == 0:
            print("Iteration {}.".format(i))
            print(mu_ts[i,])
            print(Sigma_ts[i,])
            print(pi_ts[i, ])
            print(sigma2_ts[i])
    
    return {'pi': pi_ts, 'mu' : mu_ts, 'Sigma' : Sigma_ts, 'beta': beta_ts,
            'r': r_ts, 'sigma2':sigma2_ts, 'mm1': mm1, 'rmse': rmse_ts}
    

K = 2

em_run = run_mm1_em(10, X, y, K, debug_r = debug_r[:800,:], debug_beta = mm1_sim.beta[:800,:],
                    debug_mu = mm1_sim.mu, debug_Sigma = mm1_sim.Sigma,
                    debug_pi = mm1_sim.pi)


    
