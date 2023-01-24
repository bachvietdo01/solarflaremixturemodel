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

class SolarFlareMM2REM:
    def __init__(self, AR, X, y, K, W = None, sigma20 = None, beta0 = None, pi0 = None,
                 debug_tau = None, debug_pi = None, debug_beta = None, debug_sigma2 = None):
        self.R = AR
        self.uR = np.sort(np.unique(AR))
        self.Nr = len(self.uR)
        self.X = X # covariates
        self.y = y # responses
        
        self.N = X.shape[0] # num of data points
        self.D = X.shape[1] # data dimensional
        self.K = K # number of cluster components for beta
        
        D = self.D
        
        # mapping Active Region Number to 0,..., Nr
        self.ARtor = {}
        for r in range(self.Nr):
            self.ARtor[self.uR[r]] = r
        
        # create model parameters
        self.beta = np.full((K, D), 1.0)
        self.sigma2 = np.full(K, 1.0)
        self.pi = np.full(K, 1.0/K)
        
        for k in range(K):
            self.beta[k,] = np.random.uniform(low= -10, high= 0, size=(D,))
            
        # Weighted Matrix
        if W is None:
            self.W = np.identity(self.N)
        else:
            self.W = W
        
        # initalize parameters        
        if sigma20 is not None:
            self.sigma2 = sigma20
        
        if pi0 is not None:
            self.pi = pi0
            
        if beta0 is not None:
            self.beta = beta0
        
        # debug setup
        if debug_pi is not None:
            self.pi = debug_pi    
            self.debug_pi = debug_pi
            
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
        
        self.tau = np.zeros((Nr, K))
        
        # Posterior expectation of z
        for r in range(Nr):            
            idr = (self.R == self.uR[r])
            Xr, yr = self.X[idr,], self.y[idr]
            
            for k in range(K):            
                self.tau[r,k] = np.log(self.pi[k,])
                
                for i in range(Xr.shape[0]):
                    self.tau[r,k] += dist.multivariate_normal.logpdf(yr[i], 
                                                mean =  Xr[i,].dot(self.beta[k,]), 
                                                cov = self.sigma2[k] / self.W[i,i]) 
                    
        # apply log-sum-exp traick for numerical stability
        self.tau = np.exp(self.tau - self.tau.max(axis = 1, keepdims = True))
        self.tau = self.tau / self.tau.sum(axis = 1, keepdims = True)
        
    
    def M_step(self):
        K, Nr, D = self.K, self.Nr, self.D
        
        beta_hat = np.zeros((K, D))
        sigma2_hat = np.zeros(K)
        pi_hat = np.zeros(K)
        
        # MLE for pi
        pi_hat = self.tau.sum(axis = 0) / Nr
                        
        # MLE for beta
        beta_ds = np.zeros((K, D, D))
        beta_us = np.zeros((K, D))
        
        for r in range(Nr):
            idr = (self.R == self.uR[r])
            Xr, yr = self.X[idr,], self.y[idr]
            Wr = np.diag(self.W[idr,idr])
            
            for k in range(K):                
                if self.tau[r,k] < 1e-4:
                    self.tau[r,k] = 1e-4
                    
                beta_ds[k,] += self.tau[r,k] * Xr.T.dot(Wr).dot(Xr)
                beta_us[k,] += self.tau[r,k] * yr.T.dot(Wr).dot(Xr)
        
        for k in range(K):
            beta_hat[k,] = inv(beta_ds[k,]).dot(beta_us[k,])
           
        for k in range(K):
            nk = 0
            for r in range(Nr):
                idr = (self.R == self.uR[r])
                Xr, yr = self.X[idr,], self.y[idr]
                Wr = np.diag(self.W[idr,idr])
                
                nk += Xr.shape[0] * self.tau[r,k]
                
                resrk = yr - Xr.dot(beta_hat[k,])
                sigma2_hat[k] += self.tau[r,k] * resrk.dot(Wr).dot(resrk)
                
                #for i in range(Xr.shape[0]):
                    #sigma2_hat[k] += self.tau[r,k] * np.square(yr[i] - Xr[i,].dot(beta_hat[k,]))
            
            sigma2_hat[k] /= nk
        
        self.pi = pi_hat
        self.sigma2 = sigma2_hat 
        self.beta = beta_hat
        
    def EM_iter(self):
        self.E_step()
        self.M_step()
        
    def compute_rmse(self, R_test, X_test, y_test):
        rmse = 0
        Nt = X_test.shape[0]
        
        for i in range(Nt):
             xi = X_test[i,]
             
             if not R_test[i] in self.ARtor:
                 yi_p = 0
                 
                 for k in range(self.K):
                     yi_p += self.pi[k] * xi.dot(self.beta[k,])
                 continue
             
             ri = self.ARtor[R_test[i]]
             
             yi_p = 0
             for k in range(self.K):
                 yi_p += self.tau[ri,k] * xi.dot(self.beta[k,])
             
             rmse += np.square(yi_p - y_test[i])
        
        return np.sqrt(rmse/ Nt)
    
    
    def compute_tau_4_unireg(self, R_test):
        uRt = np.unique(R_test)
        Nt = len(uRt)
        
        tau = np.zeros((Nt,self.K))
        
        for i, r in enumerate(uRt):
            ri = self.ARtor[r]
            tau[i, ] = self.tau[ri,] 
        
        return tau
    
    def compute_beta_4_unireg(self, R_test):
        uRt = np.unique(R_test)
        Nt = len(uRt)
        
        betas = np.zeros((Nt, self.D))
        
        for i, r in enumerate(uRt):
            ri = self.ARtor[r]
            
            for k in range(self.K):
                 betas[i,] += self.tau[ri,k] * self.beta[k,]
        
        return betas
    
    
    def compute_model_metric(self):
        pd = np.zeros((self.N, self.K))
        
        for k in range(self.K):
            musk = self.X.dot(self.beta[k,])
            pd[:,k] = self.pi[k] * dist.norm.pdf(self.y, musk, self.sigma2[k])
        
        ll = np.sum(np.log(pd.sum(axis = 1)))
        
        params = self.K * self.D + self.K + (self.K - 1) # betas + sigma2 + pi
        aic = -2 *ll + 2 * params
        bic = -2 * ll + params * np.log(self.N)
        
        return ll, aic, bic
    
    
    def get_responsibilites(self, R):
        res = np.zeros((len(R),self.K))
        
        for i in range(len(R)):
            ri = self.ARtor[R[i]]
            res[i,] = self.tau[ri,]
        
        return res
        
    def predict_y(self, R, X):
        N = X.shape[0]
        y = np.zeros(N)
        
        
        for i in range(N):
             xi = X[i,]
             
             # if has no information about the region take average
             if not R[i] in self.ARtor:
                 for k in range(self.K):
                     y[i] +=  self.pi[k] * xi.dot(self.beta[k,])
                 
                 continue
             
             ri = self.ARtor[R[i]]
             
             for k in range(self.K):
                 y[i] +=  self.tau[ri,k] * xi.dot(self.beta[k,])
        
        return y
    
    def predict_zi(self, R):
        z = np.zeros((len(R),self.K))
        
        for i in range(len(R)):
            ri = self.ARtor[R[i]]
            z[i,] = self.tau[ri,]
        
        return z
    
    def get_crude_zi(self):
        z = np.zeros((self.N, self.K))
        
        for i in range(self.N):
            ri = self.ARtor[self.R[i]]
            z[i,] = self.tau[ri,]
        
        return z
    
    def get_refined_zi(self):
        z = np.zeros((self.N, self.K))
        
        for i in range(self.N):
            taur = self.tau[self.ARtor[self.R[i]],]
            
            for k in range(self.K):
                if taur[k] < 1e-2:
                    taur[k] += 1e-2
                
                z[i, k] = np.log(taur[k]) + dist.multivariate_normal.logpdf(self.y[i], 
                                                mean =  self.X[i,].dot(self.beta[k,]), 
                                                cov = self.sigma2[k]) 
        
        # log-sum-exp
        z = np.exp(z - z.max(axis = 1, keepdims = True))
        z = z / z.sum(axis = 1, keepdims = True)
        
        return z
    
    def get_ri(self):
        r = np.zeros(self.N)
        
        for i in range(self.N):
            r[i] = np.where(self.uR == self.R[i])[0][0]
            
        return r
    
    def hack_refine_rmse(self):
        r = self.get_refined_zi()
        yp = np.zeros(self.N)
        
        for i in range(self.N):
            for k in range(self.K):                    
                yp[i] += r[i, k] * self.X[i,].dot(self.beta[k,])
        
        return np.sqrt(np.mean(np.square(self.y - yp)))
        
        
    def hack_rmse(self):
        yp = np.zeros(self.N)
        
        z = self.get_crude_zi().argmax(axis = 1)
        
        for k in range(self.K):
            Xk = self.X[z==k,]
            yk = self.y[z==k]
            
            betak = np.linalg.inv(Xk.T.dot(Xk)).dot(Xk.T).dot(yk)
            yp += np.sum(z == k) / self.N * self.X.dot(betak)
        
        return np.sqrt(np.mean(np.square(self.y - yp)))
    
    def hack_predict_ytrain(self):
        r = self.get_refined_zi()
        yp = np.zeros(self.N)
        
        for i in range(self.N):
            for k in range(self.K):                    
                yp[i] += r[i, k] * self.X[i,].dot(self.beta[k,])
        
        return yp
    
    def hack_y_train(self):
        yp = np.zeros(self.N)
        
        z = self.get_zi().argmax(axis = 1)
        
        for k in range(self.K):
            Xk = self.X[z==k,]
            yk = self.y[z==k]
            
            betak = np.linalg.inv(Xk.T.dot(Xk)).dot(Xk.T).dot(yk)
            yp += np.sum(z == k) / self.N * self.X.dot(betak)
        
        return yp
        
        
                
    