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
import numpy as np
import scipy.stats as dist
from numpy.linalg import inv

class SolarFlareMM2AdEM:
    def __init__(self, AR, X, y, K, W = None, sigma20 = None, beta0 = None, pir0 = None,
                 debug_tau = None, debug_beta = None, debug_sigma2 = None):
        # store active regions
        self.R = AR
        self.uR = np.sort(np.unique(AR)) # unique names
        self.Nr = len(self.uR) # total number of regions
        
        # mapping Active Region Nmae to 0,..., Nr
        self.ARtor = {}
        for r in range(self.Nr):
            self.ARtor[self.uR[r]] = r
        
        # store data
        self.X = X # covariates
        self.y = y # responses
        
        self.N = X.shape[0] # num of data points
        self.D = X.shape[1] # data dimensional
        self.K = K # number of cluster components for beta
        D = self.D
        
        # create model parameters
        self.beta = np.full((K, D), 1.0)
        self.sigma2 = np.full(K, 1.0)
        self.pir = np.full((self.Nr, K), 1.0/K)
        
        # store weighted matrix
        if W is None:
            self.W = np.identity(self.N)
        else:
            self.W = W
        
        for k in range(K):
            self.beta[k,] = np.random.uniform(low= -10, high= 0, size=(D,))
        
        # initalize parameters        
        if sigma20 is not None:
            self.sigma2 = sigma20
            
        if beta0 is not None:
            self.beta = beta0
            
        if pir0 is not None:
            self.pir = pir0
            
        # debug setup            
        if debug_sigma2 is not None:
            self.sigma2 = debug_sigma2
            self.debug_sigma2 = debug_sigma2
            
        if debug_beta is not None:
            self.beta = debug_beta
            self.debug_beta = debug_beta
        
        if debug_tau is not None:
            self.tau = debug_tau
            self.debug_rtau= debug_tau
        else:
            self.tau = np.full((self.Nr, K), 1.0 /K)
        
        # selection criteria
        self.logll = 0
        
        
    def E_step(self):
        K, Nr = self.K, self.Nr
        
        self.taui = np.zeros((self.N, K))
        self.tau = [0 for _ in range(Nr)]
        
        # Posterior expectation of z
        for r in range(Nr):    
            idr = (self.R == self.uR[r])
            Xr, yr = self.X[idr,], self.y[idr]
            
            self.tau[r] = np.zeros((np.sum(idr), K))
            
            for k in range(K):  
                for i in range(self.tau[r].shape[0]):
                    if self.pir[r, k] < 1e-8:
                        self.pir[r, k] += 1e-8
                    
                    self.tau[r][i,k] = np.log(self.pir[r, k])
                    self.tau[r][i,k] += dist.multivariate_normal.logpdf(yr[i], 
                                                mean =  Xr[i,].dot(self.beta[k,]), 
                                                cov = self.sigma2[k] / self.W[i,i]) 
                    
            # apply log-sum-exp traick for numerical stability
            self.tau[r] = np.exp(self.tau[r] - self.tau[r].max(axis = 1, keepdims = True))
            self.tau[r] = self.tau[r] / self.tau[r].sum(axis = 1, keepdims = True)
            
            # store global responsiblities
            self.taui[idr, ] = self.tau[r]
            
    
    def M_step(self):
        K, Nr, D = self.K, self.Nr, self.D
                
        beta_hat = np.zeros((K, D))
        sigma2_hat = np.zeros(K)
        pir_hat = np.zeros((Nr, K))
        
        # MLE for pi
        for r in range(Nr):
            pir_hat[r,] = self.tau[r].mean(axis = 0) 
                        
        # MLE for beta
        beta_ds = np.zeros((K, D, D))
        beta_us = np.zeros((K, D))
        
        for r in range(Nr):
            idr = (self.R == self.uR[r])
            Xr, yr = self.X[idr,], self.y[idr]
            Wr = np.diag(self.W[idr,idr])
            
            for k in range(K):
                self.tau[r][:,k] += 1e-8
                Wrk = np.multiply(np.diag(self.tau[r][:,k]), Wr)
                beta_ds[k,] += Xr.T.dot(Wrk).dot(Xr)
                beta_us[k,] += yr.T.dot(Wrk).dot(Xr)
        
        for k in range(K):
            beta_hat[k,] = inv(beta_ds[k,]).dot(beta_us[k,])
                
        
        # MLE for sigma2
        for k in range(K):
            nk = 0
            
            for r in range(Nr):
                idr = (self.R == self.uR[r])
                Xr, yr = self.X[idr,], self.y[idr]
                Wr = np.diag(self.W[idr,idr])
                
                nk += np.sum(self.tau[r][:,k])
                
                Wrk = np.multiply(Wr, np.diag(self.tau[r][:,k]))
                resrk = yr - Xr.dot(beta_hat[k,])
                sigma2_hat[k] += resrk.dot(Wrk).dot(resrk)
                
            
            sigma2_hat[k] /= nk
        
        self.pir = pir_hat
        self.sigma2 = sigma2_hat 
        self.beta = beta_hat
        
    def EM_iter(self):
        self.E_step()
        self.M_step()
        
    def compute_rmse(self, R_test, X_test, y_test):
        rmse = 0
        Nt = X_test.shape[0]
        self.z_test = np.zeros(Nt)
        
        for i in range(Nt):
             xi = X_test[i,]
             
             if not R_test[i] in self.ARtor:
                 yi_p = 0
                 for k in range(self.K):
                     yi_p +=  1./ self.K * xi.dot(self.beta[k,])
                 continue
             
             
             ri = self.ARtor[R_test[i]]
             yi_p = 0
             for k in range(self.K):
                 yi_p +=  self.pir[ri, k] * xi.dot(self.beta[k,])
             
             rmse += np.square(yi_p - y_test[i])
        
        return np.sqrt(rmse/ Nt)
    
    def predict_y(self, R, X):
        N = X.shape[0]
        y = np.zeros(N)
        
        
        for i in range(N):
             xi = X[i,]
             
             # if has no information about the region take average
             if not R[i] in self.ARtor:
                 for k in range(self.K):
                     y[i] +=  1. / self.K * xi.dot(self.beta[k,])
                 continue
             
             
             ri = self.ARtor[R[i]]
             
             for k in range(self.K):
                 y[i] +=  self.pir[ri, k] * xi.dot(self.beta[k,])
        
        return y
    
    def fit_ytrain(self):
        N = self.X.shape[0]
        y = np.zeros(N)
        
        
        for i in range(N):             
             for k in range(self.K):
                 y[i] +=  self.taui[i, k] * self.X[i,].dot(self.beta[k,])
        
        return y
            
        
        
                
    