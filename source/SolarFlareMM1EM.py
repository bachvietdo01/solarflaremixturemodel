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

class SolarFlareMM1EM:
    def __init__(self, X, y, K, sigma2_0 = 1.0, pi_0 = None, mu_0 = None, Sigma_0 = None,
                 debug_r = None, debug_beta = None, debug_pi = None, debug_mu = None, debug_Sigma = None):
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
        self.sigma2 = sigma2_0
        
        # initalize parameters
        if pi_0 is not None:
            self.pi = pi_0
        else:
            self.pi = np.full(K, 1.0 / K)
            self.pi[K-1] = 1 - np.sum(self.pi[:K - 1])
        
        if mu_0 is not None:
            self.mu = mu_0
            
        if Sigma_0 is not None:
            self.Sigma = Sigma_0
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
            
        
        # Latent variables
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
            
    def E_step_gibbs(self):
        K, N = self.K, self.N
        
        # z | beta
        self.r = np.zeros((N, K))
        
        for n in range(N):
            for k in range(K):
                self.r[n, k] = np.log(self.pi[k]) +  dist.multivariate_normal.logpdf(self.beta[n,], 
                   mean = self.mu[k,], cov = self.Sigma[k,])
                
        # apply log-sum-exp traick for numerical stability
        self.r = np.exp(self.r - self.r.max(axis = 1, keepdims = True))
        self.r = self.r / self.r.sum(axis = 1, keepdims = True)
        
        # beta | z
        # Posterior expeaction of beta
        for n in range(N):
            zn = int(np.random.choice(a = range(self.K), p = self.r[n, ], size = 1)[0])
            
            Sigmak_inv = inv(self.Sigma[zn,])
            Sigmai_tilde_inv = Sigmak_inv + np.outer(self.X[n,], self.X[n,]) / self.sigma2
            Sigmai_tilde = inv(Sigmai_tilde_inv)
            
            mui_tilde = np.matmul(Sigmak_inv, self.mu[zn, ]) + self.y[n] * self.X[n,] / self.sigma2
            mui_tilde = np.matmul(Sigmai_tilde, mui_tilde)
            
            self.beta[n, ] = mui_tilde
        
        
    def E_step(self):
        K, D, N = self.K, self.D, self.N
        
        Sigma_inv = np.zeros((K, D, D))
        Lambda_til = np.zeros((N, K, D, D))
        Lambda_til_inv = np.zeros((N, K, D, D))
        mtil = np.zeros((N, K, D))
        
        self.r = np.zeros((N, K))
        
        # Posterior expectation of z
        for k in range(self.K):
            Sigma_inv[k, ] = inv(self.Sigma[k,])
            
            for i in range(self.N):
                Lambda_til_inv[i,k, ] = (np.outer(self.X[i,], self.X[i,]) + Sigma_inv[k,]) / self.sigma2
                Lambda_til[i, k, ] = inv(Lambda_til_inv[i,k, ])
                
                mtil[i, k, ] = self.y[i] * self.X[i,] + Sigma_inv[k,].dot(self.mu[k,])   
                mtil[i, k, ] = Lambda_til[i, k, ].dot(mtil[i, k, ])
                
                    
                self.r[i,k] = np.log(self.pi[k]) + 0.5 * np.log(np.abs(det(Lambda_til[i, k, ]))) - \
                                     0.5 * np.log(np.abs(det(self.Sigma[k,]))) +  \
                                     0.5 * (-1.0 * self.mu[k, ].T.dot(Sigma_inv[k, ]).dot(self.mu[k,]) + \
                                            mtil[i,k,].T.dot(Lambda_til_inv[i,k,]).dot(mtil[i,k,]))
        
        # apply log-sum-exp traick for numerical stability
        self.r = np.exp(self.r - self.r.max(axis = 1, keepdims = True))
        self.r = self.r / self.r.sum(axis = 1, keepdims = True)
        
        # Posterior expeaction of beta
        for i in range(N):
            self.beta[i,] = np.zeros(D)
            for k in range(self.K):
                self.beta[i, ] = self.r[i,k] * mtil[i, k, ]
        
    
    def M_step(self):
        K, D, N = self.K, self.D, self.N
        
        pi_hat = np.zeros(K)
        mu_hat = np.zeros((K, D))
        Sigma_hat = np.zeros((K, D, D))
        Sigma_hat_inv = np.zeros((K, D, D))
        sigma2_hat = 0
        
        # MLE for pi, mu and Sigma
        for k in range(K):
            nk = 0
            
            for n in range(N):
                nk += self.r[n,k]
                
                pi_hat[k] += self.r[n,k]
                mu_hat[k, ] += self.r[n,k] * self.beta[n,]
            
            mu_hat[k,] /= nk
        
        pi_hat /= N
        
        for k in range(K):         
            nk = 0
            for n in range(N):
                nk += self.r[n,k]
                beta_bar = self.beta[n, ] - mu_hat[k,]
                Sigma_hat[k, ] += self.r[n,k] * np.outer(beta_bar, beta_bar)
            
            Sigma_hat[k, ] /= (self.sigma2 * nk)
            Sigma_hat_inv[k, ] = inv(Sigma_hat[k,])
        
        # MLE for sigma2                
        for n in range(N):
            sigma2_hat += np.square(self.y[n] - self.X[n,].dot(self.beta[n,]))
            
            for k in range(K):
                betank_bar = self.beta[n, ] - self.mu[k,]
                sigma2_hat += self.r[n, k] * betank_bar.T.dot(Sigma_hat_inv[k,]).dot(betank_bar)
                
        sigma2_hat /= (N + N * D) 
            
        self.pi = pi_hat
        self.mu = mu_hat
        self.sigma2 = sigma2_hat
        
        for k in range(K):
            if np.linalg.cond(Sigma_hat[k,]) < 1.0 /sys.float_info.epsilon:
                self.Sigma[k, ] = Sigma_hat[k, ]  
            
    def EM_iter(self):
        #self.E_step()
        self.M_step()
    