#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:18:20 2021

@author: vietdo
"""

import pickle as pkl
import matplotlib.pyplot as plt
import scipy.stats as dist
import pandas as pd
import numpy as np
from SolarFlareMM0EM import SolarFlareMM0EM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Load raw data
X_train_data = pd.read_csv('fulldata/X_train.csv', header = None).to_numpy()
X_test_data = pd.read_csv('fulldata/X_test.csv', header=None).to_numpy()

y_train = pd.read_csv('fulldata/y_train.csv').to_numpy()[:,0]
y_test = pd.read_csv('fulldata/y_test.csv').to_numpy()[:,0]

# Projected into PCA directions
pca = PCA(n_components=7)

X_train_data = pca.fit_transform(X_train_data)
X_test_data = pca.fit_transform(X_test_data)


# Plot PCA
fig = plt.figure(1, figsize=(10, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

ax.scatter(X_train_data[:, 0], X_train_data[:, 1], X_train_data[:, 2], 
           c = y_train, cmap=plt.cm.nipy_spectral,edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.show()


plt.scatter(X_train_data[:, 0], X_train_data[:, 1], c = y_train.tolist())
plt.xlabel('component 1')
plt.ylabel('component 2')



# Create design matrix
X_train = np.ones( (X_train_data.shape[0], X_train_data.shape[1] + 1))
X_train[:,1:] = X_train_data
X_test = np.ones( (X_test_data.shape[0], X_test_data.shape[1] + 1))
X_test[:,1:] = X_test_data

def run_mm0_em(niters, X, y, K, X_test = None, y_test = None, mm = None,
               debug_r = None, debug_beta = None, debug_sigma2 = None, debug_pi = None,
               debug_mu = None, debug_Sigma = None, pi0 = None, mu0 = None):
    D = X.shape[1]
        
    # tracking model parameters
    pi_ts = np.zeros((niters, K))
    beta_ts = np.zeros((niters, K, D))
    sigma2_ts = np.zeros((niters,K))
    mu_ts = np.zeros((niters, K, D))
    Sigma_ts = np.zeros((niters, K, D, D))
    rmse_ts = np.zeros(niters)
    logll_ts = np.zeros(niters)
    aic_ts = np.zeros(niters)
    bic_ts = np.zeros(niters)
    ecll_ts = np.zeros(niters)
    
    
    if mm is None:
        mm0 = SolarFlareMM0EM(X, y, K, debug_sigma2 = debug_sigma2,
                              debug_pi = debug_pi, debug_r = debug_r, debug_beta = debug_beta,
                              debug_mu = debug_mu, debug_Sigma = debug_Sigma, pi0 = pi0,
                              mu0 = mu0)
    else:
        mm0 = mm

    for i in range(niters):    
        mm0.EM_iter()
        mm0.compute_selection_cretia()
        
        rmse_ts[i] = mm0.compute_rmse(X_test, y_test)
        pi_ts[i, ] = mm0.pi
        beta_ts[i,] = mm0.beta
        sigma2_ts[i,] = mm0.sigma2
        mu_ts[i, ] = mm0.mu
        Sigma_ts[i,] = mm0.Sigma
        logll_ts[i] = mm0.logll
        aic_ts[i] = mm0.aic
        bic_ts[i] = mm0.bic
        ecll_ts[i] = mm0.ecll

        if i % 5 == 0:
            print("Iteration {}.".format(i))
            print(beta_ts[i,])
            print(sigma2_ts[i])
            print(pi_ts[i, ])
            print(mu_ts[i])
            print(Sigma_ts[i])
            print("rmse is {}".format(rmse_ts[i]))
            print("Expected Complete likehood is {}".format(ecll_ts[i]))
            print("Log likehood is {}".format(logll_ts[i]))
            print("AIC is {}".format(aic_ts[i]))
            print("BIC is {}".format(bic_ts[i]))
        
    
    return {'pi': pi_ts, 'beta': beta_ts, 'sigma2':sigma2_ts, 'mm0': mm0, 'mu': mu_ts,
            'Sigma': Sigma_ts, "log_ll": logll_ts, "aic": aic_ts, "bic": bic_ts,
            'ecll': ecll_ts, 'rmse': rmse_ts}

# Linear Regerssion MLE
beta_hat = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
y_pred = X_test.dot(beta_hat)
rmse = np.sqrt(np.sum(np.square(y_pred - y_test)) / y_test.shape[0])
print("Linear Regerssion RMSE is {}".format(rmse))

# Run models
K = 20

saved_model_path = 'saved_models'

kmean = KMeans(n_clusters=K, random_state=0).fit(X_train)
mu0 = kmean.cluster_centers_

mu0_min = np.amin(X_train, axis = 0)
mu0_max = np.amax(X_train, axis = 0)
for k in range(K):
    mu0[k,] = mu0_min + np.random.uniform() * (mu0_max - mu0_min)


pi0 = dist.dirichlet.rvs(np.full(K, 1))[0]
em_run = run_mm0_em(100, X_train, y_train, K, X_test, y_test, pi0 = pi0, 
                    mu0 = mu0)


f = open(saved_model_path + "/pcamm0k" + str(K) + ".obj","wb")
pkl.dump(em_run,f)
f.close()


# load model
f = open(saved_model_path + "/mm0k" + str(K) + ".obj","rb")
em_mod = pkl.load(f)
f.close()


em_run['pi'][-1]
em_run['log_ll'][-1]
em_run['aic'][-1]
em_run['bic'][-1]

# pi_hat
pi_hat = em_run['pi'][-1]
cl_nums = range(len(beta_hat[0]))
plt.bar(cl_nums, beta_hat[0], align='center', alpha=0.5)
plt.xticks(cl_nums, cl_nums)

# visualize beta cofficients
beta_hat = em_run['beta'][-1]


feat_nums = range(len(beta_hat[0]))
plt.bar(feat_nums, beta_hat[0], align='center', alpha=0.5)
plt.xticks(feat_nums, feat_nums)
plt.bar(feat_nums, beta_hat[1], align='center', alpha=0.5)
plt.xticks(feat_nums, feat_nums)
plt.bar(feat_nums, beta_hat[2], align='center', alpha=0.5)
plt.xticks(feat_nums, feat_nums)
plt.bar(feat_nums, beta_hat[3], align='center', alpha=0.5)
plt.xticks(feat_nums, feat_nums)
plt.bar(feat_nums, beta_hat[4], align='center', alpha=0.5)
plt.xticks(feat_nums, feat_nums)
