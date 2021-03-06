#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:41:38 2021

@author: vietdo
"""

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

class SolarFlareMM1pSim:  
    def __init__(self, N, D, K,mu, Sigma, sigma2, gamma):        
        self.N = N
        self.K = K
        self.D = D
        
        self.sigma2 = sigma2
        self.mu = mu
        self.Sigma = Sigma
        self.gamma = gamma
        
        self.beta = np.zeros((N, D))
        self.z = np.zeros(N)
        self.X = np.zeros((N, D))
        self.y = np.zeros(N)
        
        
    def generate(self):
        N, K, D = self.N, self.K, self.D
        
        for i in range(N):
            self.X[i,] = dist.multivariate_normal.rvs(mean = np.zeros(D), 
                                                      cov = 10 * np.identity(D), size = 1)
            
            pi = np.ones(K)
            for k in range(K-1):
                pi[k] = np.exp(self.gamma[k,].dot(self.X[i,]))
            
            pi /= sum(pi)
            
            zi = np.random.choice(a = range(self.K), p = pi, size = 1)[0]
            self.beta[i,]= dist.multivariate_normal.rvs(mean = self.mu[zi,],
                                                        cov = self.Sigma[zi,], size = 1)
            
            self.z[i] = zi
            self.y[i] = dist.multivariate_normal.rvs(mean = np.dot(self.X[i,], self.beta[i,]), 
                  cov = self.sigma2, size = 1)