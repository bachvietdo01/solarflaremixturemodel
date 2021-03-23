"""
Created on Mon Feb 22 17:18:20 2021

@author: vietdo
"""

import matplotlib.pyplot as plt
import scipy.stats as dist
import pandas as pd
import numpy as np
from SolarFlareMM0DiagEM import SolarFlareMM0DiagEM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

def run_mm0_diag_em(niters, X, y, K, X_test = None, y_test = None, mm = None,
               debug_r = None, debug_beta = None, debug_sigma2 = None, debug_pi = None,
               debug_mu = None, debug_Sd = None, pi0 = None, mu0 = None):
    D = X.shape[1]
        
    # tracking model parameters
    pi_ts = np.zeros((niters, K))
    beta_ts = np.zeros((niters, K, D))
    sigma2_ts = np.zeros((niters,K))
    mu_ts = np.zeros((niters, K, D))
    Sigmad_ts = np.zeros((niters, K, D))
    rmse_ts = np.zeros(niters)
    logll_ts = np.zeros(niters)
    aic_ts = np.zeros(niters)
    bic_ts = np.zeros(niters)
    ecll_ts = np.zeros(niters)
    
    
    if mm is None:
        mm0 = SolarFlareMM0DiagEM(X, y, K, debug_sigma2 = debug_sigma2,
                              debug_pi = debug_pi, debug_r = debug_r, debug_beta = debug_beta,
                              debug_mu = debug_mu, debug_Sigma_diag = debug_Sd, 
                              pi0 = pi0, mu0 = mu0)
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
        Sigmad_ts[i,] = mm0.Sigma_diag
        logll_ts[i] = mm0.logll
        aic_ts[i] = mm0.aic
        bic_ts[i] = mm0.bic
        ecll_ts[i] = mm0.ecll

        if i % 10 == 0:
            print("Iteration {}.".format(i))
            print(beta_ts[i,])
            print(sigma2_ts[i])
            print(pi_ts[i, ])
            print(mu_ts[i])
            print(Sigmad_ts[i])
            print("rmse is {}".format(rmse_ts[i]))
            print("Expected Complete likehood is {}".format(ecll_ts[i]))
    
    return {'pi': pi_ts, 'beta': beta_ts, 'sigma2':sigma2_ts, 'mm0': mm0, 'mu': mu_ts,
            'Sigma': Sigmad_ts, "log_ll": logll_ts, "aic": aic_ts, "bic": bic_ts,
            'ecll': ecll_ts, 'rmse': rmse_ts}

def benchmark_performance(data_path, nclusters = 2, niters = 20, em_iters = 5):
    rmse_lr = np.zeros(niters)
    rmse_mm0 = np.zeros(niters)
    bic_mm0 = np.zeros(niters)
    aic_mm0 = np.zeros(niters)


    for i in range(niters):
        X_train, X_test, y_train, y_test = load_run_data(data_path, subsample=1500)
    
        # Linear Regerssion MLE
        beta_hat = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        y_pred = X_test.dot(beta_hat)
        rmse_lr[i] = np.sqrt(np.sum(np.square(y_pred - y_test)) / y_test.shape[0])
    
        # Run models
        K = nclusters
        kmean = KMeans(n_clusters=K, random_state=0).fit(X_train)
        mu0 = kmean.cluster_centers_
    
        pi0 = dist.dirichlet.rvs(np.full(K, 1))[0]
        em_run = run_mm0_diag_em(em_iters, X_train, y_train, K, X_test, y_test, pi0 = pi0,
                    mu0 = mu0)
    
        rmse_mm0[i] = em_run['rmse'][-1]
        aic_mm0 = em_run['aic'][-1]
        bic_mm0 = em_run['bic'][-1]
    
    return rmse_lr, rmse_mm0, aic_mm0, bic_mm0


def run_EMMM_model_on_data(data_path, nclusters = 2, em_iters = 50):
    X_train, X_test, y_train, y_test = load_run_data(data_path)
    
    # Run models
    K = nclusters
    kmean = KMeans(n_clusters=K, random_state=0).fit(X_train)
    mu0 = kmean.cluster_centers_
    pi0 = dist.dirichlet.rvs(np.full(K, 1))[0]
    
    em_run = run_mm0_diag_em(em_iters, X_train, y_train, K, X_test, y_test, pi0 = pi0,
                    mu0 = mu0)
    
    return em_run



Xtrain_path  = '../fulldata/Xtrain6h.csv'
Xtest_path = '../fulldata/Xtest6h.csv'
ytrain_path  = '../fulldata/ytrain6h.csv'
ytest_path = '../fulldata/ytest6h.csv'


rmses = np.zeros(10)
for i in range(10):
    X_train, y_train, X_test, y_test = load_run_data(Xtrain_path, ytrain_path, 
                                                 Xtest_path, ytest_path, 1000, 250)
    
    K = 2
    kmean = KMeans(n_clusters=K, random_state=0).fit(X_train)
    mu0 = kmean.cluster_centers_
    pi0 = dist.dirichlet.rvs(np.full(K, 1))[0]
    em_run0 = run_mm0_diag_em(50, X_train, y_train, K, X_test, y_test, pi0 = pi0,
                    mu0 = mu0)
    rmses[i] = em_run0['rmse'][-1]

# Visualize Result
fit = em_run0['mm0']

z_fit = fit.r.argmax(axis=1)
plt.scatter(X_train[:, 1], y_train , c=z_fit, s=50, cmap='viridis')
plt.show()

plt.scatter(X_train[:, 1], X_train[:, 2] , c=z_fit, s=50, cmap='viridis')
plt.show()


# pi_hat
pi_hat = mm0_fit['pi'][-1]
cl_nums = range(len(pi_hat))
plt.bar(cl_nums, pi_hat, align='center', alpha=0.5)
plt.xticks(cl_nums, cl_nums)


# visualize mu, Sigma
x_grid = np.linspace(-10, 20, 100)
y_grid = np.linspace(-10, 20, 100)

X, Y = np.meshgrid(x_grid, y_grid)
Z1 = np.zeros((100,100))
Z2 = np.zeros((100,100))

mu1 = fit.mu[0,1:3]
mu2 = fit.mu[1,1:3]

Sigma1 = np.diag(fit.Sigma_diag[0,1:3])
Sigma2 = np.diag(fit.Sigma_diag[1,1:3])

for i in range(100):
    for j in range(100):
        Z1[i,j] = dist.multivariate_normal.pdf([X[i,j], Y[i,j]], 
                                     mean= mu1, cov= Sigma1)
        Z2[i,j] = dist.multivariate_normal.pdf([X[i,j], Y[i,j]], 
                                     mean= mu2, cov= Sigma2)


plt.xlim(-10, 20)
plt.ylim(-10, 20)
plt.contour(X, Y, Z1, colors='black')
plt.contour(X, Y, Z2, colors='blue')
plt.scatter(fit.X[:,1], fit.X[:,2])
plt.xlabel("X[,0]")
plt.ylabel("X[,1]")
plt.plot(mu1[0],mu1[1],'ro') 
plt.plot(mu2[0],mu2[1],'ro') 


# visualize y fiited
fit.sigma2 # sigma2

y_fit = fit.predict_y(fit.X)

nsize = 200
plt.plot(range(len(y_fit[:nsize])), y_fit[:nsize], color = 'red')  
plt.plot(range(len(y_fit[:nsize])), fit.y[:nsize], color='gray')  


# fit test data
_, X_test, _, y_test = load_data(data_path)
test_fit = fit.predict_y(X_test)

nsize = 200
plt.plot(range(len(test_fit[:nsize])), test_fit[:nsize], color = 'blue')  
plt.plot(range(len(test_fit[:nsize])), y_test[:nsize], color='gray')  


# visualize beta cofficients
beta_hat = mm0_fit['beta'][-1]

feat_nums = range(len(beta_hat[0,1:3]))
plt.bar(feat_nums, beta_hat[0,1:3], align='center', alpha=0.5)
plt.xticks(feat_nums, feat_nums)


plt.bar(feat_nums, beta_hat[1,1:3], align='center', alpha=0.5)
plt.xticks(feat_nums, feat_nums)

