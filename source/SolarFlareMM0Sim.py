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

class SolarFlareMM0Sim:  
    def __init__(self, N, D, K, mu, Sigma, pi, beta, sigma2):        
        self.N = N
        self.K = K
        self.D = D
        
        self.pi = pi
        self.mu = mu
        self.Sigma = Sigma
        self.sigma2 = sigma2
        self.beta = beta
        
        self.z = np.zeros(N)
        self.X = np.zeros((N, D))
        self.y = np.zeros(N)
        
        
    def generate(self):
        
        for i in range(self.N):
            z = np.random.choice(a = range(self.K), p = self.pi, size = 1)[0]
            self.z[i] = z
            self.X[i,] = dist.multivariate_normal.rvs(mean = self.mu[z, ], cov = self.Sigma[z,], size = 1)
            self.y[i] = dist.multivariate_normal.rvs(mean = np.dot(self.X[i,], self.beta[z,]), 
                  cov = self.sigma2[z], size = 1)