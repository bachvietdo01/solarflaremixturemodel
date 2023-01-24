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

class SolarFlareMM2AdSim:  
    def __init__(self, N, Nr, D, K, beta, sigma2):        
        self.N = N
        self.Nr = Nr
        self.K = K
        self.D = D
        
        self.sigma2 = sigma2
        self.beta = beta
        
        self.z = np.zeros(N)
        self.pir = np.zeros((Nr, K))
        self.X = np.zeros((N, D))
        self.y = np.zeros(N)
        self.R = np.zeros(N)
        
        
    def generate(self):
        N, Nr, K, D = self.N, self.Nr, self.K, self.D
        
        
        # generate region assignment
        self.R = np.random.randint(0, Nr, N)
        
        # generate mixture propotion for each region
        for r in range(Nr):
            self.pir[r,] = dist.dirichlet.rvs([1.0/ K] * K)
        
        # generate data for each region
        for i in range(N):
            self.X[i,] = dist.multivariate_normal.rvs(mean = np.zeros(D), 
                                                      cov = 10 * np.identity(D), size = 1)
            
            zi = int(np.random.choice(a = range(K), p = self.pir[self.R[i]], size = 1)[0])
            
            self.z[i] = zi
            self.y[i] = dist.multivariate_normal.rvs(mean = np.dot(self.X[i,], self.beta[zi,]), 
                  cov = self.sigma2[zi], size = 1)