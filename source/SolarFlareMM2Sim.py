#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:33:13 2021

@author: vietdo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:24:21 2021

@author: vietdo
"""

import numpy as np
import scipy.stats as dist

class SolarFlareMM2Sim:  
    def __init__(self, N, D, K, sigma2, pi, beta):        
        self.N = N
        self.K = K
        self.D = D
        
        self.pi = pi
        self.beta = beta
        self.sigma2 = sigma2
        
        self.z = np.zeros(N)
        self.X = np.zeros((N, D))
        self.y = np.zeros(N)
        
    """def __init__(self, N , D, K, mu0, Sigma0, a, b, alpha = None):        
        self.mu0 = mu0
        self.Sigma0 = Sigma0
        
        self.a = a
        self.b
        self.alpha = alpha
        
        if alpha is None:
            alpha = np.array([1.0 / K] * K)
            alpha[K - 1] = 1 - sum(self.pi[:K-1])
        
        self.pi = dist.dirichlet.rvs(alpha, size = 1)
        self.beta = np.zeros((K, D))
        self.sigma2= np.zeros(K)
        
        for k in range(K):
            self.sigma2[k] = dist.invgamma.rvs(a = 1, scale = 1)
            self.beta[k,] = dist.multivariate_normal.rvs(mean= mu0, cov = self.sigma2[k] * Sigma0)
        
        self.z = np.zeros((N, K))"""
        
    def generate(self):
        
        for i in range(self.N):
            z = np.random.choice(a = range(self.K), p = self.pi, size = 1)[0]
            self.z[i] = z
            self.X[i,] = dist.multivariate_normal.rvs(mean = np.zeros(self.D), cov = 100 * np.identity(self.D), size = 1)
            self.y[i] = dist.multivariate_normal.rvs(mean = np.dot(self.X[i,], self.beta[z,]), 
                  cov = self.sigma2[z], size = 1)