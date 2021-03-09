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
from numpy.linalg import inv
import numpy.linalg as linalg

class SolarFlareMM0EM:
    def __init__(self, X, y, K, sigma20 = None, pi0 = None, mu0 = None, Sigma0 = None, beta0 = None,
                 debug_r = None, debug_beta = None, debug_pi = None, debug_mu = None, 
                 debug_Sigma = None, debug_sigma2 = None):
        self.X = X # xovariates
        self.y = y # response
        
        self.N = X.shape[0] # num of data points
        self.D = X.shape[1] # data dimensional
        
        self.K = K # number of cluster components for beta
        
        N = self.N
        D = self.D
                
        
        # create model parameters
        self.pi = np.zeros(K)
        self.mu = np.zeros((K, D))
        self.Sigma = np.zeros((K, D, D))
        self.beta = np.full((K, D), 1)
        self.sigma2 = np.full(K, 1)
        
        # initalize parameters
        if pi0 is not None:
            self.pi = pi0
        else:
            self.pi = np.full(K, 1.0 / K)
            self.pi[K-1] = 1 - np.sum(self.pi[:K - 1])
        
        if mu0 is not None:
            self.mu = mu0
        
        if sigma20 is not None:
            self.sigma2 = sigma20
        
        if Sigma0 is not None:
            self.Sigma = Sigma0
        else:
            for k in range(K):
                self.Sigma[k,] = np.identity(D)
        
        # debug setup
        if debug_mu is not None:
            self.mu = debug_mu
            self.debug_mu = debug_mu
        
        if debug_Sigma is not None:
            self.Sigma = debug_Sigma
            self.debug_Sigma = debug_Sigma
        
        if debug_pi is not None:
            self.pi = debug_pi
            self.debug_pi = debug_pi
            
        if debug_sigma2 is not None:
            self.sigma2 = debug_sigma2
            self.debug_sigma2 = debug_sigma2
            
        if debug_beta is not None:
            self.beta = debug_beta
            self.debug_beta = debug_beta
        else:
          self.beta = np.zeros((N, D))
        
        if debug_r is not None:
            self.r = debug_r
            self.debug_r = debug_r
        else:
            self.r = np.full((N, K), 1.0 /K)
            
        
        # selection criteria
        self.logll = 0
        self.aic = 0
        self.bic = 0
        self.ecll = 0
        
        
    def E_step(self):
        K, N = self.K, self.N
        
        self.r = np.zeros((N, K))
        
        # Posterior expectation of z
        for k in range(self.K):            
            for i in range(self.N):
                self.r[i,k] = np.log(self.pi[k]) + dist.multivariate_normal.logpdf(self.X[i,], mean=self.mu[k,], cov= self.Sigma[k, ]) + dist.multivariate_normal.logpdf(self.y[i], mean = self.X[i, ].dot(self.beta[k,]), 
                                                cov = self.sigma2[k]) 
        
        # apply log-sum-exp traick for numerical stability
        self.r = np.exp(self.r - self.r.max(axis = 1, keepdims = True))
        self.r = self.r / self.r.sum(axis = 1, keepdims = True)
        
    
    def M_step(self):
        K, D, N = self.K, self.D, self.N
        
        pi_hat = np.zeros(K)
        mu_hat = np.zeros((K, D))
        Sigma_hat = np.zeros((K, D, D))
        beta_hat = np.zeros((K, D))
        sigma2_hat = np.zeros(K)
        
        # MLE for pi, mu
        for k in range(K):
            nk = 0
            
            for n in range(N):
                nk += self.r[n,k]
                
                pi_hat[k] += self.r[n,k]
                mu_hat[k,] += self.r[n,k] * self.X[n,]
            
            mu_hat[k,] /= nk
        
        pi_hat /= N
        
        # MLE for Sigma
        for k in range(K):         
            nk = 0
            for n in range(N):
                nk += self.r[n,k]
                xn_bar = self.X[n, ] - mu_hat[k,]
                Sigma_hat[k, ] += self.r[n,k] * np.outer(xn_bar, xn_bar)
            
            Sigma_hat[k, ] /= nk
        
        # MLE for beta                
        for k in range(K):
            betak_ds = np.zeros((D, D))
            betak_us = np.zeros(D)
            
            for n in range(N):
                if self.r[n,k] < 1e-4:
                    self.r[n,k] += 1e-4
                
                betak_ds += self.r[n, k] * np.outer(self.X[n,], self.X[n,])
                betak_us += self.r[n, k] * self.y[n] * self.X[n, ]
            
            if linalg.cond(betak_ds) > 1.0/sys.float_info.epsilon:
                print(self.r[n, k])
                print('debug')
            
            beta_hat[k,] = inv(betak_ds).dot(betak_us)
        
        # MLE for sigma2
        for k in range(K):
            nk = 0
            
            for n in range(N):
                nk += self.r[n,k]
                sigma2_hat[k] += self.r[n,k] * np.square(self.y[n] - self.X[n,].dot(self.beta[k,]))
            
            sigma2_hat[k] /= nk
        
        
        self.pi = pi_hat
        self.mu = mu_hat
        self.sigma2 = sigma2_hat
        self.beta = beta_hat
        
        for k in range(K):
            if np.linalg.cond(Sigma_hat[k,]) < 1.0 /sys.float_info.epsilon:
                self.Sigma[k, ] = Sigma_hat[k, ]  
            
    def EM_iter(self):
        self.E_step()
        self.M_step()
        
    def compute_rmse(self, X_test, y_test):
        rmse = 0
        Nt = X_test.shape[0]
        self.z_test = np.zeros(Nt)
        
        for i in range(Nt):
             xi = X_test[i,]
             yi_p = 0
             ri = np.zeros(self.K)
             
             for k in range(self.K):
                 ri[k] = np.log(self.pi[k]) + dist.multivariate_normal.logpdf(xi, 
                             mean=self.mu[k,], cov= self.Sigma[k, ])
                 
             ri = np.exp(ri - np.max(ri))
             ri = ri / np.sum(ri)
             zi = np.argmax(ri)
             self.z_test[i] = zi
             
             yi_p = xi.dot(self.beta[zi,])
             
             rmse += np.square(yi_p - y_test[i])
        
        return np.sqrt(rmse/ Nt)
    
    def compute_selection_cretia(self):
        N, K = self.N, self.K
        
        zx = np.zeros((N, K))
        
        for k in range(K):
            for n in range(N):
                if linalg.cond(self.Sigma[k,]) < 1.0 /sys.float_info.epsilon:
                    self.Sigma[k,] += 1e-4 * np.identity(self.D)
                
                zx[n, k] = np.log(self.pi[k]) + dist.multivariate_normal.logpdf(self.X[n,],
                                                 mean=self.mu[k,], cov = self.Sigma[k,])
        
        zx = np.exp(zx - zx.max(axis = 1, keepdims = True))
        zx = zx / zx.sum(axis = 1, keepdims = True)
        
        logll = 0
        ecll = 0
        
        for n in range(N):
            li = 0
            for k in range(K):
                # add to log likelihood
                li += zx[n,k] * dist.multivariate_normal.pdf(self.X[n,],
                                                 mean=self.mu[k,], cov = self.Sigma[k,])
                
                # add to expected complete log likelihood
                ecll += self.r[n,k] * ( dist.multivariate_normal.logpdf(self.X[n,],
                                                 mean=self.mu[k,], cov = self.Sigma[k,]) + \
                                        dist.multivariate_normal.logpdf(self.y[n,],
                                                 mean=self.X[n,].dot(self.beta[k,]), 
                                                 cov = self.sigma2[k]) + np.log(self.pi[k]))
                             
                              
            
            logll += np.log(li)
        
        # compute model selection crtieria
        self.logll = logll
        self.ecll = ecll
        
        parm_no = 4 * K * self.D + K * self.D * (self.D + 1) / 2.0
        self.aic = -2 * logll + 2 * parm_no
        self.bic = -2 * logll  + parm_no * np.log(N)
                
    