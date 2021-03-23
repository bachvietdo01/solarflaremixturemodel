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

class SolarFlareMM2pEM:
    def __init__(self, X, y, K, sigma20 = None, beta0 = None, gamma0 = None,
                 debug_r = None, debug_gamma = None, debug_beta = None, debug_sigma2 = None):
        self.X = X # covariates
        self.y = y # responses
        
        self.N = X.shape[0] # num of data points
        self.D = X.shape[1] # data dimensional
        self.K = K # number of cluster components for beta
        
        N = self.N
        D = self.D
        
        # create model parameters
        self.gamma = np.full((K-1, D), 1.0)
        self.beta = np.full((K, D), 1.0)
        self.sigma2 = np.full(K, 1.0)
        
        for k in range(K):
            self.beta[k,] = np.random.uniform(low= -1, high= 1, size=(D,))
            
            if k < K -1:
                 self.gamma[k,] = np.random.uniform(low= -1, high= 1, size=(D,))
        
        # initalize parameters        
        if sigma20 is not None:
            self.sigma2 = sigma20
        
        if gamma0 is not None:
            self.gamma = gamma0
        
        # debug setup
        if debug_gamma is not None:
            self.gamma = debug_gamma    
            self.debug_gamma = debug_gamma
            
        if debug_sigma2 is not None:
            self.sigma2 = debug_sigma2
            self.debug_sigma2 = debug_sigma2
            
        if debug_beta is not None:
            self.beta = debug_beta
            self.debug_beta = debug_beta
        
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
        for i in range(self.N):
            # for 1 ... K -1
            for k in range(self.K - 1):            
                self.r[i,k] = self.gamma[k,].dot(self.X[i,]) + dist.multivariate_normal.logpdf(self.y[i], mean = self.X[i, ].dot(self.beta[k,]), 
                                                cov = self.sigma2[k]) 
                
            # for K
            piK = 1 
            self.r[i,K-1] = np.log(piK) + dist.multivariate_normal.logpdf(self.y[i], mean = self.X[i, ].dot(self.beta[K-1,]), 
                                                cov = self.sigma2[K-1]) 
        
        # apply log-sum-exp traick for numerical stability
        self.r = np.exp(self.r - self.r.max(axis = 1, keepdims = True))
        self.r = self.r / self.r.sum(axis = 1, keepdims = True)
        
    
    def M_step(self):
        K, D, N = self.K, self.D, self.N
        
        beta_hat = np.zeros((K, D))
        sigma2_hat = np.zeros(K)
                
        # MLE for beta                
        for k in range(K):
            betak_ds = np.zeros((D, D))
            betak_us = np.zeros(D)
            
            for n in range(N):
                if self.r[n,k] < 1e-4:
                    self.r[n,k] += 1e-4
                
                betak_ds += self.r[n, k] * np.outer(self.X[n,], self.X[n,])
                betak_us += self.r[n, k] * self.y[n] * self.X[n, ]
            
            beta_hat[k,] = inv(betak_ds).dot(betak_us)
        
        # MLE for sigma2
        for k in range(K):
            nk = 0
            
            for n in range(N):
                nk += self.r[n,k]
                sigma2_hat[k] += self.r[n,k] * np.square(self.y[n] - self.X[n,].dot(self.beta[k,]))
            
            sigma2_hat[k] /= nk
        
        self.sigma2 = sigma2_hat 
        self.beta = beta_hat
        self.gamma = self.grad_descent_for_gamma() # us gradient descent for gamma
    
    def grad_descent_for_gamma(self, tol = 1e-1, max_iters = 100, rho = 0.1):
        def obj_fn(gamma):
            K, N = self.K, self.N
            
            val = 0
            for n in range(N):
                normalized_const = 0
                for k in range(K-1):
                    val += self.r[n,k] * gamma[k,].dot(self.X[n,])
                    normalized_const += np.exp(gamma[k,].dot(self.X[n,]))
                
                val += np.log(1 + normalized_const)
            
            return val
                
        K = self.K
        gamma_last = gamma_now = np.zeros((K - 1, self.D))
        
        obj_now, obj_last = np.inf, 0
        iters = 0
        
        while np.abs(obj_now - obj_last) > tol and iters < max_iters:
            iters += 1
            
            gamma_last = gamma_now
            obj_last = obj_now
            
            # do gradient descent
            for k in range(K-1):
                grad = 0
                
                for i in range(self.N):                    
                    pi = np.zeros(K)
                    
                    for j in  range(K-1):
                        pi[j] = gamma_last[j,].dot(self.X[i,])
                        
                    pi = np.exp(pi - np.max(pi))
                    pi = pi / np.sum(pi)
                        
                    grad += (self.r[i,k] - pi[k]) * self.X[i,]
                
                grad = grad / np.sqrt(np.sum(np.square(grad))) # normalized gradient
                gamma_now[k,] = gamma_last[k,] + rho * grad
        
            obj_now = obj_fn(gamma_now)
        
        print("Number of iterations is {}.".format(iters))
            
                
        return gamma_now
            
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
             for k in range(self.K - 1):
                 ri[k] = self.gamma[k,].dot(X_test[i,])
                 
             ri = np.exp(ri - np.max(ri))
             ri = ri / np.sum(ri)
             
             zi = np.argmax(ri)
             self.z_test[i] = zi
             
             yi_p = xi.dot(self.beta[zi,])
             
             rmse += np.square(yi_p - y_test[i])
        
        return np.sqrt(rmse/ Nt)
    
    def predict_y(self, X):
        N = X.shape[0]
        y = np.zeros(N)
        
        
        for i in range(N):
             xi = X[i,]
             
             ri = np.zeros(self.K)
             for k in range(self.K - 1):
                 ri[k] = self.gamma[k,].dot(xi)
                 
             ri = np.exp(ri - np.max(ri))
             ri = ri / np.sum(ri)
             
             zi = np.argmax(ri)
             
             y[i] = xi.dot(self.beta[zi,])
        
        return y
                
    