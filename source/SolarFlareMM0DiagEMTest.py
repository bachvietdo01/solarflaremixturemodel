#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 15:46:35 2021

@author: vietdo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:45:48 2021

@author: vietdo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:42:40 2021

@author: vietdo
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:45:34 2021

@author: vietdo
"""

import scipy.stats as dist
import matplotlib.pyplot as plt
import numpy as np
from SolarFlareMM0Sim import SolarFlareMM0Sim
from SolarFlareMM0DiagEM import SolarFlareMM0DiagEM

# simulate data
N = 1000
D = 2
K = 2

beta = np.zeros((K, D))
sigma2 = np.zeros(K)
mu = np.zeros((K, D))
Sigma = np.zeros((K, D, D))


pi = np.array([0.8, 0.2])
beta[0,] = [-1 ,0]
beta[1,] = [1, 0]
sigma2[0] = 1
sigma2[1] = 1
mu[0,] = [-5, 0]
mu[1,] = [5, 0]
Sigma[0,] = 3 * np.identity(D)
Sigma[1,] = 2 * np.identity(D)

mm0_sim = SolarFlareMM0Sim(N, D, K, sigma2 = sigma2, pi = pi, beta = beta,
                           mu = mu, Sigma = Sigma)
mm0_sim.generate()

# initalize Solar Flare MM1
X = mm0_sim.X[:800,]
y = mm0_sim.y[:800,]
X_test = mm0_sim.X[800:,]
y_test = mm0_sim.y[800:,]

plt.scatter(X[:,0], X[:,1])
plt.xlabel("X[,0]")
plt.ylabel("X[,1]")

plt.scatter(X[:,0], y)
plt.xlabel("X[,0]")
plt.xlabel("y")

# create design matrix
# X_design = np.ones( (X.shape[0], X.shape[1] + 1))
# X_design[:,1:] = X

# X_test_design = np.ones( (X_test.shape[0], X_test.shape[1] + 1))
# X_test_design[:,1:] = X_test

# X = X_design
# X_test = X_test_design

debug_r = np.zeros((N, D))

for n in range(N):
    if mm0_sim.z[n] == 1:
        debug_r[n, 1] = 1
    else:
        debug_r[n, 0] = 1


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

        if i % 1 == 0:
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
            'ecll': ecll_ts}
    

# Linear Regerssion MLE
beta_hat = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
y_pred = X_test.dot(beta_hat)
rmse = np.sqrt(np.sum(np.square(y_pred - y_test)) / y_test.shape[0])
print("Linear Regerssion RMSE is {}".format(rmse))


K = 2

# create a random initalization point
mu0 = np.zeros((K, X.shape[1]))

mu0_min = np.amin(X, axis = 0)
mu0_max = np.amax(X, axis = 0)

for k in range(K):
    mu0[k,] = mu0_min + np.random.uniform() * (mu0_max - mu0_min)


pi0 = dist.dirichlet.rvs(np.full(K, 1))[0]
em_run = run_mm0_diag_em(100, X, y, K, X_test, y_test, pi0 = pi0,
                    mu0 = mu0)



# plot out log like hood trace
ecll_ts = em_run['ecll']
ll_ts = em_run['log_ll']

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

axes[0].plot(range(len(ecll_ts)), ecll_ts)
axes[0].set_title("Expected Complete likelihood trace")
axes[0].set_ylabel("Likehood")
axes[0].set_xlabel("Iteration")

axes[1].plot(range(len(ll_ts)), ll_ts)
axes[1].set_title("Log likelihood trace")
axes[1].set_ylabel("Likehood")
axes[1].set_xlabel("Iteration")

fig.tight_layout()

# plot out sigma2 trace
sigma2_ts = em_run['sigma2'][:35,]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

axes[0].plot(range(len(sigma2_ts[:,0])), sigma2_ts[:,0])
axes[0].set_title("Cluster 1 sigma2 trace")
axes[0].set_ylabel("sigma2")
axes[0].set_xlabel("Iteration")

axes[1].plot(range(len(sigma2_ts[:,1])), sigma2_ts[:,1])
axes[1].set_title("Cluster 2 sigma2 trace")
axes[1].set_ylabel("sigma2")
axes[1].set_xlabel("Iteration")

fig.tight_layout()

# plot out beta trace
beta_ts = em_run['beta'][:35,]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

axes[0,0].plot(range(len(beta_ts[:,0, 0])), beta_ts[:,0, 0])
axes[0,0].set_title("Cluster 1 beta_0 trace")
axes[0,0].set_ylabel("beta[0,0]")
axes[0,0].set_xlabel("Iteration")

axes[0,1].plot(range(len(beta_ts[:,0, 1])), beta_ts[:,0, 1])
axes[0,1].set_title("Cluster 1 beta_1 trace")
axes[0,1].set_ylabel("beta[0,1]")
axes[0,1].set_xlabel("Iteration")

axes[1,0].plot(range(len(beta_ts[:,1, 0])), beta_ts[:,1, 0])
axes[1,0].set_title("Cluster 2 beta_0 trace")
axes[1,0].set_ylabel("beta[1,0]")
axes[1,0].set_xlabel("Iteration")

axes[1,1].plot(range(len(beta_ts[:,1, 1])), beta_ts[:,1, 1])
axes[1,1].set_title("Cluster 2 beta_1 trace")
axes[1,1].set_ylabel("beta[1,1]")
axes[1,1].set_xlabel("Iteration")

fig.tight_layout()

# plot out pi trace
pi_ts = em_run['pi'][:35,]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

axes[0].plot(range(len(pi_ts[:,0])), pi_ts[:,0])
axes[0].set_title("Cluster 1 mixing weight")
axes[0].set_ylabel("pi[0]")
axes[0].set_xlabel("Iteration")

axes[1].plot(range(len(pi_ts[:,1])), pi_ts[:,1])
axes[1].set_title("Cluster 2 mixing weight")
axes[1].set_ylabel("pi[1]")
axes[1].set_xlabel("Iteration")

fig.tight_layout()

# plot out mu trace
mu_ts = em_run['mu'][:35,]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

axes[0,0].plot(range(len(mu_ts[:,0, 0])), mu_ts[:,0, 0])
axes[0,0].set_title("Cluster 1 mux trace")
axes[0,0].set_ylabel("mu[0,0]")
axes[0,0].set_xlabel("Iteration")

axes[0,1].plot(range(len(mu_ts[:,0,1])), mu_ts[:,0, 1])
axes[0,1].set_title("Cluster 1 my trace")
axes[0,1].set_ylabel("mu[0,1]")
axes[0,1].set_xlabel("Iteration")

axes[1,0].plot(range(len(mu_ts[:,1, 0])), mu_ts[:,1, 0])
axes[1,0].set_title("Cluster 2 mux trace")
axes[1,0].set_ylabel("mu[1,0]")
axes[1,0].set_xlabel("Iteration")

axes[1,1].plot(range(len(mu_ts[:,1,1])), mu_ts[:,1, 1])
axes[1,1].set_title("Cluster 2 my trace")
axes[1,1].set_ylabel("mu[1,1]")
axes[1,1].set_xlabel("Iteration")

fig.tight_layout()

# plot out Sigma trace
Sigma_ts = em_run['Sigma'][:35,]


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

axes[0,0].plot(range(len(Sigma_ts[:,0, 0])), Sigma_ts[:,0, 0])
axes[0,0].set_title("Cluster 1 Sigma11 trace")
axes[0,0].set_ylabel("sigma2")
axes[0,0].set_xlabel("Iteration")

axes[0,1].plot(range(len(Sigma_ts[:,0, 1])), Sigma_ts[:,0, 1])
axes[0,1].set_title("Cluster 1 Sigma22 trace")
axes[0,1].set_ylabel("sigma2")
axes[0,1].set_xlabel("Iteration")


axes[1,0].plot(range(len(Sigma_ts[:,1, 0])), Sigma_ts[:,1, 0])
axes[1,0].set_title("Cluster 2 Sigma11 trace")
axes[1,0].set_ylabel("sigma2")
axes[1,0].set_xlabel("Iteration")

axes[1,1].plot(range(len(Sigma_ts[:,1, 1])), Sigma_ts[:,1, 1])
axes[1,1].set_title("Cluster 2 Sigma22 trace")
axes[1,1].set_ylabel("sigma22")
axes[1,1].set_xlabel("Iteration")

fig.tight_layout()


K = 10

pi0 = dist.dirichlet.rvs(np.full(K, 1))[0]
em_run = run_mm0_em(50, X, y, K, X_test, y_test, pi0 = pi0)

em_run['aic'][-1]
em_run['bic'][-1]
em_run['log_ll'][-1]


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

# pi_hat
pi_hat = em_run['pi'][-1]
cl_nums = range(len(pi_hat))
plt.bar(cl_nums, pi_hat, align='center', alpha=0.5)
plt.xticks(cl_nums, cl_nums)




    
