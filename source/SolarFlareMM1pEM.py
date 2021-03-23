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

class SolarFlareMM1pEM:
    def __init__(self, X, y, K, sigma20 = None, beta0 = None, mu0 = None, Sigma0 = None, gamma0 = None,
                 debug_mu = None, debug_Sigma = None, debug_beta = None, debug_sigma2 = None, 
                 debug_gamma = None, debug_r = None):
        self.X = X # covariates
        self.y = y # responses
        
        self.N = X.shape[0] # num of data points
        self.D = X.shape[1] # data dimensional
        self.K = K
        
        N = self.N
        D = self.D
        
        # create model parameters
        self.mu = np.zeros((K, D))
        self.beta = np.full((N, D), 1.0)
        self.sigma2 = 1
        self.Sigma = np.zeros((K, D, D))
        self.gamma = np.full((K-1, D), 1.0)
        self.r = np.zeros((N, K))
        
        for k in range(K):
            self.Sigma[k,] = np.identity(D)
        
        # initalize parameters        
        if sigma20 is not None:
            self.sigma2 = sigma20
        
        if mu0 is not None:
            self.mu = mu0
        
        if Sigma0 is not None:
            self.Sigma = Sigma0
            
        if beta0 is not None:
            self.beta = beta0
            
        if gamma0 is not None:
            self.gamma = gamma0
        
        # debug setup
        if debug_mu is not None:
            self.mu = debug_mu  
            self.debug_mu = debug_mu
        
        if debug_Sigma is not None:
            self.Sigma = debug_Sigma  
            self.debug_Sigma = debug_Sigma
            
        if debug_sigma2 is not None:
            self.sigma2 = debug_sigma2
            self.debug_sigma2 = debug_sigma2
            
        if debug_beta is not None:
            self.beta = debug_beta
            self.debug_beta = debug_beta
            
        if debug_gamma is not None:
            self.beta = debug_gamma
            self.debug_gamma = debug_gamma
            
        if debug_r is not None:
            self.r = debug_r
            self.debug_r = debug_r
        
        # selection criteria
        self.logll = 0
        self.aic = 0
        self.bic = 0
        self.ecll = 0
        
        
    def E_step(self):
        K, N, D = self.K, self.N, self.D
        
        self.r = np.zeros((N, K))
        
        # Posterior expectation of z[n] | X[n,], y[n,]
        # for 1 ... K -1
        for k in range(K - 1):    
            for i in range(N):
                xi = self.X[i,]
                
                Sigmak_inv = inv(self.Sigma[k,])
                
                Gammaik = inv(np.outer(xi, xi) / self.sigma2 + Sigmak_inv)
                
                tauik = 1 / self.sigma2 - xi.dot(Gammaik).dot(xi)
                tauik = 1/ tauik
                
                mik = tauik * xi.dot(Gammaik).dot(Sigmak_inv.dot(self.mu[k,].T))
                
                if k != K -1:
                    self.r[i,k] = self.gamma[k,].dot(self.X[i,]) 
                else:
                    self.r[i,k] = 0 # log(1)
                
                if tauik < 0:
                    tauik = 1e-4
                
                self.r[i,k]  += dist.multivariate_normal.logpdf(self.y[i], mean = mik, 
                                                cov = tauik) 
                
        
        # apply log-sum-exp traick for numerical stability
        self.r = np.exp(self.r - self.r.max(axis = 1, keepdims = True))
        self.r = self.r / self.r.sum(axis = 1, keepdims = True)
        
        # posterior expectation of beta[n,] | X[n,], y[n]
        for n in range(N):
            self.beta[n,] = np.zeros(D)
            
            for k in range(K):
                Sigmak_inv = inv(self.Sigma[k,])
                Lambdank = np.outer(self.X[n,], self.X[n,]) / self.sigma2 + Sigmak_inv
                
                self.beta[n,] += self.r[n,k] * inv(Lambdank).dot(Sigmak_inv.dot(self.mu[k,]) + 
                                                                self.y[n]* self.X[n,])
        
    
    def M_step(self):
        D, N, K = self.D, self.N, self.K
        
        sigma2_hat = 0
        mu_hat = np.zeros((K, D))
        Sigma_hat = np.zeros((K, D, D))
                
        # MLE for mu, Sigma
        for k in range(K):
            nk = 0
            for n in range(N):
                nk += self.r[n, k]
                mu_hat[k,] += self.r[n,k] * self.beta[n,]
            
            mu_hat[k,] /=nk
        
        for k in range(K):
            nk = 0
            for n in range(N):
                nk += self.r[n, k]
                betank_cen = self.beta[n,] - mu_hat[k,]
                Sigma_hat[k,] += self.r[n, k] * np.outer(betank_cen, betank_cen)
            
            Sigma_hat[k,] /= nk

        
        # MLE for sigma2
        for n in range(N):
            sigma2_hat += np.square(self.y[n] - self.X[n,].dot(self.beta[n,]))
        sigma2_hat /= N
        
        self.sigma2 = sigma2_hat 
        self.mu = mu_hat
        self.Sigma = Sigma_hat
        self.gamma = self.grad_descent_for_gamma() # gradient descent for gamma
    
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
        gamma_last = gamma_now = np.zeros((K-1, self.D))
        
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
             
             betai = np.zeros(self.D)
             for k in range(self.K):
                 betai = ri[k] * self.mu[k,]
                 
             yi_p = xi.dot(betai)
             
             rmse += np.square(yi_p - y_test[i])
        
        return np.sqrt(rmse/ Nt)
    
    def predict_y(self, X):
        N = X.shape[0]
        y = np.zeros(N)
        
        
        for i in range(N):
             xi = X[i,]
             
             ri = np.ones(self.K)
             for k in range(self.K - 1):
                 ri[k] = self.gamma[k,].dot(xi)
                 
             ri = np.exp(ri - np.max(ri))
             ri = ri / np.sum(ri)
             
             betai = np.zeros(self.D)
             for k in range(self.K):
                 betai = ri[k] * self.mu[k,]
             
             y[i] = xi.dot(betai)
        
        return y
                
    