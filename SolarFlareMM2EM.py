#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 21:28:45 2021

@author: vietdo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:43:55 2021

@author: vietdo
"""
import sys
import numpy as np
import scipy.stats as dist
from numpy.linalg import inv, det

class SolarFlareMM2EM:
    def __init__(self, X, y, K, sigma2_0 = 1.0, pi_0 = None, beta_0 = None,
                 debug_r = None, debug_beta = None, debug_pi = None, debug_sigma2 = None):
        
        self.tol = 1e-4 # tolerance for updating sigma2
        self.X = X # xovariates
        self.y = y # response
        
        self.N = X.shape[0] # num of data points
        self.D = X.shape[1] # data dimensional
        self.K = K # number of cluster components for beta
        
        N = self.N
        D = self.D
                
        
        # create model parameters
        self.pi = np.zeros(K)
        self.beta = np.zeros((K, D))
        
        # initalize parameters
        if pi_0 is not None:
            self.pi = pi_0
        else:
            self.pi = np.full(K, 1.0 / K)
            self.pi[K-1] = 1 - np.sum(self.pi[:K - 1])
        
        if beta_0 is not None:
            self.beta = beta_0
            
        if sigma2_0 is not None:
            self.sigma2 = np.full(K, sigma2_0)
        else:
            for k in range(K):
                self.sigma2[k] = 1.0
        
        # debug setup
        if debug_beta is not None:
            self.beta = debug_beta
            self.debug_beta = debug_beta
            
        for k in range(K):
            self.beta[k,] += dist.multivariate_normal.rvs(mean = np.zeros(D), cov = 10 * np.identity(D))
        
        if debug_sigma2 is not None:
            self.sigma2 = debug_sigma2
            self.debug_sigma2 = debug_sigma2
        
        if debug_pi is not None:
            self.pi = debug_pi
            self.debug_pi = debug_pi
            
        # Latent variables
        if debug_r is not None:
            self.r = debug_r
            self.debug_r = debug_r
        else:
          self.r = np.full((N, K), 1.0 /K)
        
    def E_step(self):
        K, N = self.K, self.N
                
        self.r = np.zeros((N, K))
        
        # Posterior expectation of z
        for k in range(K):            
            for i in range(N):
                self.r[i,k] = np.log(self.pi[k]) + dist.multivariate_normal.logpdf(self.y[i], 
                   mean = self.X[i,].dot(self.beta[k,]), cov = self.sigma2[k])
        
        # apply log-sum-exp traick for numerical stability
        self.r = np.exp(self.r - self.r.max(axis = 1, keepdims = True))
        self.r = self.r / self.r.sum(axis = 1, keepdims = True)
        
    
    def M_step(self):
        K, D, N = self.K, self.D, self.N
        
        pi_hat = np.zeros(K)
        beta_hat = np.zeros((K, D))
        sigma2_hat = np.zeros(K)
        
        # MLE for pi, mu and sigma2
        for k in range(K):
            betahk_dn = np.zeros((D,D))
            nk = 0
            
            for n in range(N):
                nk += self.r[n,k]
                
                pi_hat[k] += self.r[n,k]
                beta_hat[k, ] += self.r[n,k] * self.X[n,] * self.y[n]
                betahk_dn += self.r[n,k] * np.outer(self.X[n,], self.X[n, ])
            
            beta_hat[k,] = inv(betahk_dn).dot(beta_hat[k,])
        
        pi_hat /= N
        
        for k in range(K):         
            nk = 0
            
            for n in range(N):
                nk += self.r[n,k]
                sigma2_hat[k] += self.r[n,k] * np.square(self.y[n] - self.X[n,].dot(beta_hat[k,]))
            
            sigma2_hat[k] /= nk
        
        self.pi = pi_hat
        self.beta = beta_hat
        
        print(sigma2_hat)
        for k in range(K):
            if sigma2_hat[k] > self.tol:
                self.sigma2[k] = sigma2_hat[k]
                
            
    def EM_iter(self):
        self.E_step()
        self.M_step()
        
    def compute_rmse(self, X_test, y_test):
        rmse = 0
        Nt = X_test.shape[0]
            
        for i in range(Nt):
             zi = int(np.random.choice(a = range(self.K), p = self.pi, size = 1)[0])
             yi_p = dist.norm.rvs(loc = X_test[i,].dot(self.beta[zi,]), 
                                    scale = np.square(self.sigma2[zi]), size = 1)
             rmse += np.square(yi_p - y_test[i])
             
             if np.isnan(rmse):
                 print('debug')
        
        return np.sqrt(rmse/ Nt)
    
    def compute_rmse_alt(self, X_test, y_test):
        rmse = 0
        Nt = X_test.shape[0]
            
        for i in range(Nt):
             yi_p = 0
             
             for k in range(self.K):
                 yi_p += self.pi[k] * X_test[i,].dot(self.beta[k,])
             
             rmse += np.square(yi_p - y_test[i])
        
        return np.sqrt(rmse/ Nt)
    
    def compute_model_selection_criteria(self):
        ll = 0
        
        for n in range(self.N):
            ln = 0
            for k in range(self.K):
                ln += self.pi[k] * dist.multivariate_normal.pdf(self.y[n], 
                               mean = self.X[n,].dot(self.beta[k,]) , 
                              cov = self.sigma2[k])
                
            ll += np.log(ln)
        
        self.ll = ll
        
        nparms = self.K * (self.D + 2) - 1
        self.aic = 2*ll - nparms
        self.bic = 2*ll - nparms * np.log(self.N)
        
            
            
            
    