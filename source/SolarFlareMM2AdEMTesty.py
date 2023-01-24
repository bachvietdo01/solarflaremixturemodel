#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:46:35 2021

@author: vietdo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:45:48 2021

@author: vietdo
"""

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

import matplotlib.pyplot as plt
import numpy as np
from SolarFlareMM2AdSim import SolarFlareMM2AdSim
from SolarFlareMM2AdEM import SolarFlareMM2AdEM


def convert_to_design_mat(X):
    Xd = np.ones((X.shape[0], X.shape[1] + 1))
    Xd[:,1:] = X
    
    return Xd

# simulate data
N = 10000
Nr = 50
D = 2
K = 2

beta = np.zeros((K, D))
sigma2 = np.zeros(K)


sigma2 = [1, 1]
beta[0,] = [5 ,0]
beta[1,] = [-5, 0]

mm2Ad_sim = SolarFlareMM2AdSim(N, Nr, D, K, sigma2 = sigma2, beta = beta)
mm2Ad_sim.generate()

# initalize Solar Flare MM1
R = mm2Ad_sim.R[:8000,]
X = convert_to_design_mat(mm2Ad_sim.X[:8000,])
y = mm2Ad_sim.y[:8000,]

R_test = mm2Ad_sim.R[2000:,]
X_test = convert_to_design_mat(mm2Ad_sim.X[2000:,])
y_test = mm2Ad_sim.y[2000:,]

plt.scatter(X[:,1], X[:,2])
plt.xlabel("X[,0]")
plt.ylabel("X[,1]")
plt.show()

plt.scatter(X[:,1], y)
plt.xlabel("X[,0]")
plt.xlabel("y")
plt.show()

def run_mm2Ad_em(niters, R, X, y, K, R_test, X_test = None, y_test = None, mm = None,
               debug_tau = None, debug_beta = None, debug_sigma2 = None, 
               beta0 = None, sigma20 = None):
    D = X.shape[1]
        
    # tracking model parameters
    beta_ts = np.zeros((niters, K, D))
    sigma2_ts = np.zeros((niters,K))
    rmse_ts = np.zeros(niters)
    
    if mm is None:
        mm2Ad = SolarFlareMM2AdEM(R, X, y, K, debug_sigma2 = debug_sigma2,
                              debug_tau = debug_tau, debug_beta = debug_beta, 
                              sigma20 = sigma20, beta0 = beta0)
    else:
        mm2Ad = mm

    for i in range(niters):    
        mm2Ad.EM_iter()
        
        rmse_ts[i] = mm2Ad.compute_rmse(R_test, X_test, y_test)
        sigma2_ts[i,] = mm2Ad.sigma2
        beta_ts[i,] = mm2Ad.beta

        if i % 5 == 0:
            print("Iteration {}.".format(i))
            print(beta_ts[i,])
            print(sigma2_ts[i])
            print("rmse is {}".format(rmse_ts[i]))
    
    return {'beta': beta_ts, 'sigma2': sigma2_ts, 'mm2Ad': mm2Ad, 
            'rmse': rmse_ts}
    
    
# Linear Regerssion MLE
beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
y_pred = X_test.dot(beta_hat)
rmse = np.sqrt(np.sum(np.square(y_pred - y_test)) / y_test.shape[0])
print("Linear Regerssion RMSE is {}".format(rmse))


K = 2

beta0 = np.zeros((K, D + 1))
em_run = run_mm2Ad_em(50, R, X, y, K, R_test, X_test, y_test)













    
