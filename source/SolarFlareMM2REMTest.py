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

import scipy.stats as dist
import matplotlib.pyplot as plt
import numpy as np
from SolarFlareMM2RSim import SolarFlareMM2RSim
from SolarFlareMM2REM import SolarFlareMM2REM
from sklearn.cluster import KMeans


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
pi = np.zeros(K)


sigma2 = [1, 1]
beta[0,] = [10 ,0]
beta[1,] = [5, 0]
pi = [0.3, 0.7]

mm2R_sim = SolarFlareMM2RSim(N, Nr, D, K, sigma2 = sigma2, beta = beta, pi = pi)
mm2R_sim.generate()

# initalize Solar Flare MM1
R = mm2R_sim.R[:800,]
X = convert_to_design_mat(mm2R_sim.X[:800,])
y = mm2R_sim.y[:800,]

R_test = mm2R_sim.R[800:,]
X_test = convert_to_design_mat(mm2R_sim.X[800:,])
y_test = mm2R_sim.y[800:,]

plt.scatter(X[:,1], X[:,2])
plt.xlabel("X[,0]")
plt.ylabel("X[,1]")
plt.show()

plt.scatter(X[:,1], y)
plt.xlabel("X[,0]")
plt.xlabel("y")
plt.show()




def run_mm2R_em(niters, R, X, y, K, X_test = None, y_test = None, mm = None,
               debug_tau = None, debug_beta = None, debug_sigma2 = None, debug_pi = None,
               pi0 = None, beta0 = None, sigma20 = None):
    D = X.shape[1]
        
    # tracking model parameters
    pi_ts = np.zeros((niters, K))
    beta_ts = np.zeros((niters, K, D))
    sigma2_ts = np.zeros((niters,K))
    rmse_ts = np.zeros(niters)
    
    if mm is None:
        mm2R = SolarFlareMM2REM(R, X, y, K, debug_sigma2 = debug_sigma2,
                              debug_pi = debug_pi, debug_tau = debug_tau, 
                              debug_beta = debug_beta, sigma20 = sigma20, 
                              beta0 = beta0, pi0 = pi0)
    else:
        mm2R = mm

    for i in range(niters):    
        mm2R.EM_iter()
        
        rmse_ts[i] = mm2R.compute_rmse(X_test, y_test)
        sigma2_ts[i,] = mm2R.sigma2
        beta_ts[i,] = mm2R.beta
        pi_ts[i,] = mm2R.pi

        if i % 1 == 0:
            print("Iteration {}.".format(i))
            print(beta_ts[i,])
            print(sigma2_ts[i])
            print(pi_ts[i,])
            print("rmse is {}".format(rmse_ts[i]))
    
    return {'beta': beta_ts, 'sigma2': sigma2_ts, 'mm2R': mm2R, 
            'pi': pi_ts,'rmse': rmse_ts}
    

# Linear Regerssion MLE
beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
y_pred = X_test.dot(beta_hat)
rmse = np.sqrt(np.sum(np.square(y_pred - y_test)) / y_test.shape[0])
print("Linear Regerssion RMSE is {}".format(rmse))


K = 2

beta0 = np.zeros((K, D + 1))
em_run = run_mm2R_em(20, R, X, y, K, X_test, y_test)








    
