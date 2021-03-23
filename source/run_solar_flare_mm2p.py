#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:18:20 2021

@author: vietdo
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from SolarFlareMM2pEM import SolarFlareMM2pEM
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

mpl.style.use('ggplot')


def load_run_data(Xtrain_path, ytrain_path, Xtest_path, ytest_path, 
                  tr_size = None, ts_size = None):
    # Load raw data
    X_train_data = pd.read_csv(Xtrain_path).to_numpy()
    X_test_data = pd.read_csv(Xtest_path).to_numpy()

    y_train = pd.read_csv(ytrain_path).to_numpy()[:,0].astype('float')
    y_test = pd.read_csv(ytest_path).to_numpy()[:,0].astype('float')

    # Projected into PCA directions
    pca = PCA(n_components=3)

    X_train_data = pca.fit_transform(X_train_data)
    X_test_data = pca.fit_transform(X_test_data)
    
    # Create design matrix
    X_train = np.ones( (X_train_data.shape[0], X_train_data.shape[1] + 1))
    X_train[:,1:] = X_train_data
    X_test = np.ones( (X_test_data.shape[0], X_test_data.shape[1] + 1))
    X_test[:,1:] = X_test_data
    
    
    if tr_size is None:
        tr_size = X_train.shape[0]
    
    if ts_size is None:
        ts_size = X_test.shape[0]
        
    train_idx =np.random.choice(range(X_train.shape[0]), tr_size, replace=False)
    test_idx =np.random.choice(range(X_test.shape[0]), ts_size, replace=False)
    
    
    return X_train[train_idx,:], y_train[train_idx], X_test[test_idx,:], y_test[test_idx]


def run_mm2p_em(niters, X, y, K, X_test = None, y_test = None, mm = None,
               debug_r = None, debug_beta = None, debug_sigma2 = None, debug_gamma = None,
               gamma0 = None, beta0 = None, sigma20 = None):
    D = X.shape[1]
        
    # tracking model parameters
    gamma_ts = np.zeros((niters, K, D))
    beta_ts = np.zeros((niters, K, D))
    sigma2_ts = np.zeros((niters,K))
    rmse_ts = np.zeros(niters)
    
    if mm is None:
        mm2p = SolarFlareMM2pEM(X, y, K, debug_sigma2 = debug_sigma2,
                              debug_gamma = debug_gamma, debug_r = debug_r, 
                              debug_beta = debug_beta, sigma20 = sigma20, 
                              beta0 = beta0, gamma0 = gamma0)
    else:
        mm2p = mm

    for i in range(niters):    
        mm2p.EM_iter()
        
        rmse_ts[i] = mm2p.compute_rmse(X_test, y_test)
        sigma2_ts[i,] = mm2p.sigma2
        gamma_ts[i,] = mm2p.gamma
        beta_ts[i,] = mm2p.beta
        gamma_ts[i,] = mm2p.gamma

        if i % 1 == 0:
            print("Iteration {}.".format(i))
            print(beta_ts[i,])
            print(sigma2_ts[i])
            print(gamma_ts[i,])
            print("rmse is {}".format(rmse_ts[i]))
    
    return {'beta': beta_ts, 'sigma2': sigma2_ts, 'beta_ts':beta_ts, 'mm2p': mm2p, 
            'rmse': rmse_ts}


Xtrain_path  = '../fulldata/Xtrain6h.csv'
Xtest_path = '../fulldata/Xtest6h.csv'
ytrain_path  = '../fulldata/ytrain6h.csv'
ytest_path = '../fulldata/ytest6h.csv'

X_train, y_train, X_test, y_test = load_run_data(Xtrain_path, ytrain_path, 
                                                 Xtest_path, ytest_path, 1000, 250)


# Plot relationship between X and y in 2D
plt.scatter(X_train[:,1], y_train)
plt.show()

# Plot relationship between X and y in 3D
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(X_train[:,1], X_train[:,2], y_train)


# Plot PCA
fig = plt.figure(1, figsize=(10, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
ax.scatter(X_train[:, 1], X_train[:, 2], X_train[:, 3], 
           c = y_train, cmap=plt.cm.nipy_spectral,edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
plt.show()


plt.scatter(X_train[:, 1], X_train[:, 2], c = (y_train < -5))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.show()


rmses_lr = np.zeros(10)
rmses = np.zeros(10)
for i in range(10):
    X_train, y_train, X_test, y_test = load_run_data(Xtrain_path, ytrain_path, 
                                                 Xtest_path, ytest_path, 1000, 250)
    
    # Linear Regerssion MLE
    beta_hat = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
    y_pred = X_test.dot(beta_hat)
    rmse = np.sqrt(np.sum(np.square(y_pred - y_test)) / y_test.shape[0])
    print("Linear Regerssion RMSE is {}".format(rmse))
    rmses_lr[i] = rmse

    # Run models
    K = 2
    em_run1p = run_mm2p_em(50,  X_train, y_train, K, X_test, y_test)
    rmses[i] = em_run1p['rmse'][-1]


z_fit = em_run1p['mm2p'].r.argmax(axis=1)
plt.scatter(X_train[:, 1], y_train , c=z_fit, s=50, cmap='viridis')
plt.show()

plt.scatter(X_train[:, 1], X_train[:, 2] , c=z_fit, s=50, cmap='viridis')
plt.show()






# visualize y fiited
fit = em_run['mm2p']

y_fit = fit.predict_y(fit.X)

nsize = 200
plt.plot(range(len(y_fit[:nsize])), y_fit[:nsize], color = 'red')  
plt.plot(range(len(y_fit[:nsize])), fit.y[:nsize], color='gray')  
plt.title('Training Set: fitted y vs true y')
plt.show()

plt.scatter(fit.y[:nsize], y_fit[:nsize], color = 'red')
plt.xlabel("Truth")
plt.ylabel("Fitted")
plt.xlim((-16.5, -10))
plt.ylim((-16.5, -10))
plt.title('Training Set: fitted y vs true y')
plt.show()


# fit test data
test_fit = fit.predict_y(X_test)

nsize = 200
plt.plot(range(len(test_fit[:nsize])), test_fit[:nsize], color = 'blue')  
plt.plot(range(len(test_fit[:nsize])), y_test[:nsize], color='gray')  
plt.title('Test Set: fitted y vs true y')
plt.show()

plt.scatter(y_test[:nsize], test_fit[:nsize], color = 'blue')
plt.xlabel("Truth")
plt.ylabel("Fitted")
plt.title('Test Set: fitted y vs true y')
plt.xlim((-16.5, -10))
plt.ylim((-16.5, -10))
plt.show()




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
