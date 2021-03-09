#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:43:55 2021

@author: vietdo
"""

import numpy as np
import scipy.stats as dist
from numpy.linalg import inv

class SolarFlareMM1:
    def __init__(self, X, y, K, alpha, mu0, kappa0, Lambda0, nu0,
                 debug_mu = None, debug_Sigma = None, debug_beta = None, 
                 debug_sigma2 = None, debug_z = None):
        self.X = X # xovariates
        self.y = y # response
        
        self.N = X.shape[0] # num of data points
        self.D = X.shape[1] # data dimensional
        
        self.K = K # number of cluster components for beta
        
        N = self.N
        D = self.D
                
        # store hyperparameter
        self.alpha = alpha
        self.mu0 = mu0
        self.kappa0 = kappa0
        self.Lambda0 = Lambda0
        self.nu0 = nu0
        
        # create model parameters
        self.pi = np.zeros(K)
        self.mu = np.zeros((K, D))
        self.Sigma = np.zeros((K, D, D))
        
        self.beta = np.zeros((N, D))
        self.z = np.zeros(N).astype(int)
        self.sigma2 = 1.0
        
        # initalize model parameters
        self.pi = dist.dirichlet.rvs(alpha, size = 1)[0]
        self.sigma2 = dist.invgamma.rvs(a = 1, scale = 1)
        
        for k in range(self.K):
            self.Sigma[k, ] = dist.invwishart.rvs(df=self.nu0, scale= self.Lambda0, size = 1)
            self.mu[k,] = dist.multivariate_normal.rvs(mean = self.mu0, 
                   cov =self.Lambda0 / self.kappa0, size = 1)
            
        for n in range(self.N):
            z = np.random.choice(a = range(self.K), p = self.pi, size = 1)[0]
            self.z[n] = int(z)
            self.beta[n,] = dist.multivariate_normal.rvs(mean = self.mu[z, ] , 
                                                     cov = self.sigma2 * self.Sigma[z, ], size = 1)
        
        # assign debug variables
        self.debug_mu = debug_mu
        self.debug_Sigma = debug_Sigma
        self.debug_sigma2 = debug_sigma2
        self.debug_beta = debug_beta
        self.debug_z = debug_z
        
        
        if debug_mu is not None:
            self.mu = debug_mu
        
        if debug_Sigma is not None:
            self.Sigma = debug_Sigma
            
        if debug_sigma2 is not None:
            self.sigma2 = debug_sigma2
            
        if debug_beta is not None:
            self.beta = debug_beta
        
        if debug_z is not None:
            self.debug_z = debug_z
        
        
    def sample_beta_and_sigma2(self):  
        sh = 1.0 + self.N / 2.0
        sc = 0.0
        
        for n in range(self.N):
            zn = self.z[n]
            
            Sigmak_inv = inv(self.Sigma[zn,])
            Sigmai_tilde_inv = Sigmak_inv + np.outer(self.X[n,], self.X[n,])
            Sigmai_tilde = inv(Sigmai_tilde_inv)
            
            mui_tilde = np.matmul(Sigmak_inv, self.mu[zn, ]) + self.y[n] * self.X[n,]
            mui_tilde = np.matmul(Sigmai_tilde, mui_tilde)
            
            # sample beta[n,]
            self.beta[n, ] = dist.multivariate_normal.rvs(mean = mui_tilde, cov = self.sigma2 * Sigmai_tilde,
                size = 1)
            
            # add up the scale parameter for sigma2
            sc += np.square(self.y[n])
            sc += self.mu[zn, ].T.dot(Sigmak_inv).dot(self.mu[zn, ])
            sc -= mui_tilde.T.dot(Sigmai_tilde_inv).dot(mui_tilde)
        
        sc = 1. + sc / 2.0
        self.sigma2 = dist.invgamma.rvs(a = sh, scale = sc, size = 1)

    
    def sample_z(self):
        mat_z = np.zeros((self.N, self.K))
        
        for n in range(self.N):
            for k in range(self.K):
                mat_z[n, k] = np.log(self.pi[k]) +  dist.multivariate_normal.logpdf(self.beta[n,], 
                   mean = self.mu[k,], cov = self.sigma2 *  self.Sigma[k,])
                
        # apply log-sum-exp traick for numerical stability
        mat_z = np.exp(mat_z - mat_z.max(axis = 1, keepdims = True))
        mat_z = mat_z / mat_z.sum(axis = 1, keepdims = True)
        
        # sample z
        for n in range(self.N):
            self.z[n] =int(np.random.choice(a = range(self.K), p = mat_z[n, ], size = 1)[0])
        
        
    def sample_pi(self):
        new_alpha = np.zeros(self.K)
        
        for k in range(self.K):
            new_alpha[k] = self.alpha[k] + sum(self.z == k)
            
        self.pi = dist.dirichlet.rvs(new_alpha, size = 1)[0]
        
    def sample_mu_and_Sigma(self):
        for k in range(self.K):
            idx_k = (self.z == k)
            
            if sum(idx_k) == 0:
                continue
            
            nk = np.sum(idx_k)
            betak_bar = sum(self.beta[idx_k,]) / nk
            beta_cen = self.beta[idx_k, ] - betak_bar
            Sk = np.einsum('ij,ik->jk', beta_cen, beta_cen)
            
            mk_tilde = self.kappa0 / (self.kappa0 + nk) * self.mu0 + nk / (self.kappa0 + nk) * betak_bar
            Lambdak_tilde = self.Lambda0 + Sk + self.kappa0 * nk /(self.kappa0 + nk) * np.outer(betak_bar - self.mu0, betak_bar - self.mu0)
            kappak_tilde = self.kappa0 + nk
            nuk_tilde = self.nu0 + nk
            
            self.Sigma[k,] = dist.invwishart.rvs(nuk_tilde, Lambdak_tilde)
            self.mu[k,] = dist.multivariate_normal.rvs(mean = mk_tilde, cov = self.Sigma[k,]/ kappak_tilde)
            
    # one Gibbs iteration
    def gibbs_iter(self):
        if self.debug_z is None:
            self.sample_z()
        
        self.sample_pi()
        
        if self.debug_mu is None or self.debug_Sigma is None:
            self.sample_mu_and_Sigma()
        
        # reordering the order of mu, Sigma, pi
        so = self.mu[:,0].argsort()
        self.mu = self.mu[so,]
        self.Sigma = self.Sigma[so,]
        self.pi = self.pi[so]
        
        if self.debug_beta is None or self.debug_sigma2 is None:
            self.sample_beta_and_sigma2()
            
    # compute rmse 
    def compute_rmse(self, X_test, y_test):
        N = X_test.shape[0]
        rmse = 0.0
        
        for n in range(N):
            z = int(np.random.choice(a = range(self.K), p = self.pi, size = 1)[0])
            beta = dist.multivariate_normal.rvs(mean = self.mu[z,], cov = self.sigma2 * self.Sigma[z,], size = 1)
            
            y_p = dist.norm.rvs(loc = X_test[n,].dot(beta), scale = np.square(self.sigma2), size = 1)
            
            rmse += np.square(y_p - y_test[n])
        
        return np.sqrt(rmse / N)