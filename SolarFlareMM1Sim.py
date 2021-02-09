#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 16:24:21 2021

@author: vietdo
"""

import numpy as np
import scipy.stats as dist

class SolarFlareMM1Sim:  
    def __init__(self, N , D, K, sigma2, mu0, kappa0, Lambda0, nu0):
        self.__assign(N, D, K , sigma2)
        
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.Lambda0 = Lambda0
        self.nu0 = nu0
        
        alpha = np.array([1.0 / K] * K)
        alpha[K - 1] = 1 - sum(self.pi[:K-1])
        
        self.pi = dist.dirichlet.rvs(alpha, size = 1)
        self.mu = np.zeros((K, D))
        self.Sigma = np.zeros((K, D, D))
        
        for k in range(K):
            self.Sigma[k,] = dist.invwishart.rvs(nu0, Lambda0, 1)
            self.mu[k,] = dist.multivariate_normal.rvs(mean= mu0, cov = self.Sigma[k,] / kappa0 )
    
    def __init__(self, N, D, K, sigma2, pi, mu, Sigma):
        self.__assign(N, D, K, sigma2)
        
        self.pi = pi
        self.mu = np.zeros((K, D))
        self.Sigma = np.zeros((K, D, D))
        
        for k in range(K):
            self.Sigma[k,] = Sigma[k, ]
            self.mu[k,] = mu[k,]
            
    
    def __assign(self, N, D, K, sigma2):
        self.N = N
        self.D = D
        self.K = K
        
        self.X = np.zeros((N, D))
        self.y = np.zeros(N)
        
        self.sigma2 = sigma2
        
        self.z = np.zeros(N)
        self.beta = np.zeros((N, D))
        
    
    def generate(self):
        
        for i in range(self.N):
            z = np.random.choice(a = range(self.K), p = self.pi, size = 1)[0]
            self.z[i] = z
            self.beta[i,] = dist.multivariate_normal.rvs(mean = self.mu[z,], 
                     cov = self.sigma2 * self.Sigma[z,], size = 1)
            self.X[i,] = dist.multivariate_normal.rvs(mean = np.zeros(self.D)  , cov = np.identity(self.D), size = 1)
            self.y[i] = dist.multivariate_normal.rvs(mean = np.dot(self.X[i,], self.beta[i,]), 
                  cov = self.sigma2, size = 1)
        
        