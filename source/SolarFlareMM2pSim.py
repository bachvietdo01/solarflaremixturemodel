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

class SolarFlareMM2pSim:  
    def __init__(self, N, D, K,beta, sigma2, gamma):        
        self.N = N
        self.K = K
        self.D = D
        
        self.sigma2 = sigma2
        self.beta = beta
        self.gamma = gamma
        
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
            
            z = np.random.choice(a = range(self.K), p = pi, size = 1)[0]
            
            self.z[i] = z
            self.y[i] = dist.multivariate_normal.rvs(mean = np.dot(self.X[i,], self.beta[z,]), 
                  cov = self.sigma2[z], size = 1)