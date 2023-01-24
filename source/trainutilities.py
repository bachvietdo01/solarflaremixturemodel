#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 16:19:43 2021

@author: vietdo
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from scipy.stats import t
import pickle

# constants
colors = ["green", "orange","blue","red", "cyan", "magenta", 
          "violet", "grey", "yellowgreen", "black", "wheat",
          "teal", "azure", "plum", "pink", "honeydew"]
markes = ['o', 'x', '+', '^', 'v', 'X']


# sampling utilities
def load_run_data(path, ts = 0.25, seed_no = 0):
    data = pd.read_csv(path)
    
    X = data.iloc[:,:21].to_numpy()
    y = data.iloc[:,21].to_numpy().astype('float')
    X = np.delete(X, (2, 8, 13, 20), axis = 1)
    
    # split train and test set according to test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=seed_no)
    
    R_train = X_train[:,0]
    R_test = X_test[:,0]
    
    X_train = X_train[:,1:21]
    X_test = X_test[:,1:21]
    
    # ensure test set only contains ARs in train set
    test_diff = set(R_test).difference(set(R_train))
    test_idx = ~np.isin(R_test, list(test_diff))
    
    X_test = X_test[test_idx, :]
    R_test = R_test[test_idx]
    y_test = y_test[test_idx]
    
    
    # standardize each column of train and test separately
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    
    # create pca projection
    pca_train = PCA(n_components=7).fit(X_train)
    Xpca_train = pca_train.transform(X_train)
    pca_test= PCA(n_components=7).fit(X_test)
    Xpca_test = pca_test.transform(X_test)
    
    # Create design matrix
    X_dtrain = np.ones( (X_train.shape[0], X_train.shape[1] + 1))
    X_dtrain[:,1:] = X_train
    X_dtest = np.ones( (X_test.shape[0], X_test.shape[1] + 1))
    X_dtest[:,1:] = X_test
    
    
    return R_train, X_dtrain, y_train, R_test, X_dtest, y_test, Xpca_train, Xpca_test, pca_train, pca_test


def reblance_data(X, y, threshold = -5, ubr = 1.0):
    idxhi = (y >= threshold) # high indices
    
    Xhi, yhi = X[idxhi,], y[idxhi,]
    Xlo, ylo = X[~idxhi,], y[~idxhi,]
    
    nhi,nlo = Xhi.shape[0], Xlo.shape[0]
    
    # subsample from low index
    subidx = np.random.choice(np.arange(0,nlo), ubr * nhi, replace = False)
    Xlo, ylo = Xlo[subidx,], ylo[subidx]
    
    
    # create rebalanced data
    X = np.vstack((Xhi, Xlo))
    y = np.concatenate((yhi, ylo))
    
    return X, y

def reblance_data2(R, X, y, threshold = -5, ubr = 1.0):
    idxhi = (y >= threshold) # high indices
    
    Xhi, yhi, Rhi = X[idxhi,], y[idxhi,], R[idxhi,]
    Xlo, ylo, Rlo = X[~idxhi,], y[~idxhi,], R[~idxhi,]
    
    nhi,nlo = Xhi.shape[0], Xlo.shape[0]
    
    # subsample from low index
    subidx = np.random.choice(np.arange(0,nlo), ubr * nhi, replace = False)
    Xlo, ylo, Rlo = Xlo[subidx,], ylo[subidx], R[subidx,]
    
    
    # create rebalanced data
    X = np.vstack((Xhi, Xlo))
    y = np.concatenate((yhi, ylo))
    R = np.vstack((Rhi, Rlo))
    
    return R, X, y


def load_balanced_run_data(path, ts = 0.25, threshold = -5, seed_no = 0, ubr = 1):
    data = pd.read_csv(path) # load the data
    
    # rebalancing the high and low labels according to ubr
    X = data.iloc[:,:21].to_numpy()
    y = data.iloc[:,21].to_numpy().astype('float')
    X = np.delete(X, (2, 8, 13, 20), axis = 1)
    X, y = reblance_data(X, y, threshold = threshold, ubr = ubr)
    
    # split train and test set according to test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=seed_no)
    
    R_train = X_train[:,0]
    R_test = X_test[:,0]
    
    X_train = X_train[:,1:21]
    X_test = X_test[:,1:21]
    
    # ensure test set only contains ARs in train set
    test_diff = set(R_test).difference(set(R_train))
    test_idx = ~np.isin(R_test, list(test_diff))
    
    X_test = X_test[test_idx, :]
    R_test = R_test[test_idx]
    y_test = y_test[test_idx]
    
    
    # standardize each column of train and test separately
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)
    
    # create pca projection
    pca_train = PCA(n_components=7).fit(X_train)
    Xpca_train = pca_train.transform(X_train)
    pca_test= PCA(n_components=7).fit(X_test)
    Xpca_test = pca_test.transform(X_test)
    
    # Create design matrix
    X_dtrain = np.ones( (X_train.shape[0], X_train.shape[1] + 1))
    X_dtrain[:,1:] = X_train
    X_dtest = np.ones( (X_test.shape[0], X_test.shape[1] + 1))
    X_dtest[:,1:] = X_test
    
    
    return R_train, X_dtrain, y_train, R_test, X_dtest, y_test, Xpca_train, Xpca_test, pca_train, pca_test


def load_run_data_split_by_AR(path, ts = 0.2, vs = 0.2,
                              seed_no = 0, threshold = -5, 
                              ubr = None):
    data = pd.read_csv(path)
    
    # split train and test data under each Active Region
    D = data.shape[1] - 2
    
    X = data.iloc[:,:D].to_numpy()
    y = data.iloc[:,D].to_numpy().astype('float')
    

    X = np.delete(X, (5, 11, 16, 23), axis = 1)  
    D -= 8
    
    if ubr is not None:
        X, y = reblance_data(X, y, threshold, ubr)
    
    
    R, X = X[:,0:4],  X[:,4:]
    
    ts += vs # add valdiate size to test size
    R_train, X_train, y_train = np.zeros((1, 4)), np.zeros((1, D)), np.zeros(1)
    R_test, X_test, y_test = np.zeros((1,4)), np.zeros((1, D)), np.zeros(1)
    
    for l in np.sort(np.unique(R[:,0])):
        idl = (R[:,0] == l)
        
        Rl, Xl, yl = R[idl,:], X[idl,:], y[idl]
            
        
        if Rl.shape[0] < 4:
            R_train = np.concatenate((R_train, Rl))
            y_train = np.concatenate((y_train, yl))
            X_train = np.vstack((X_train, Xl))
        else:
            indices = np.arange(Rl.shape[0])
            Xtr, Xts, ytr, yts, idtr, idts = train_test_split(Xl, yl, indices,  test_size=ts, random_state=seed_no)
            
            R_train = np.concatenate((R_train, Rl[idtr,:]))
            y_train = np.concatenate((y_train, ytr))
            X_train = np.vstack((X_train, Xtr))
            
            R_test = np.concatenate((R_test, Rl[idts,:]))
            y_test = np.concatenate((y_test, yts))
            X_test = np.vstack((X_test, Xts))
    
    # remove the 1st dummy entry
    R_train, X_train, y_train= R_train[1:,], X_train[1:,], y_train[1:,]
    R_test, X_test, y_test = R_test[1:,], X_test[1:,], y_test[1:,]
    
    # create validation set
    test_mask = np.zeros(X_test.shape[0], bool)
    
    N_val = int(X_test.shape[0] * vs / ts)
    val_idx = np.random.choice(X_test.shape[0], size=N_val, replace=False)
    test_mask[val_idx] = True
    
    R_val, X_val, y_val = R_test[~test_mask,], X_test[~test_mask,], y_test[~test_mask,]
    R_test, X_test, y_test = R_test[test_mask,], X_test[test_mask,], y_test[test_mask,]
    
    
    # standardize each column of train and test separately
    X_train = StandardScaler().fit_transform(X_train)
    X_val = StandardScaler().fit_transform(X_val)
    X_test = StandardScaler().fit_transform(X_test)
    
    # create pca projection
    pca_train = PCA(n_components=7).fit(X_train)
    Xpca_train = pca_train.transform(X_train)
    
    pca_val = PCA(n_components=7).fit(X_val)
    Xpca_val = pca_train.transform(X_val)
    
    pca_test= PCA(n_components=7).fit(X_test)
    Xpca_test = pca_test.transform(X_test)
    
    # Create design matrix
    X_dtrain = np.ones( (X_train.shape[0], X_train.shape[1] + 1))
    X_dtrain[:,1:] = X_train
    
    X_dval = np.ones( (X_val.shape[0], X_val.shape[1] + 1))
    X_dval[:,1:] = X_val
    
    X_dtest = np.ones( (X_test.shape[0], X_test.shape[1] + 1))
    X_dtest[:,1:] = X_test  
    
    return R_train, X_dtrain, y_train, \
           R_val, X_dval, y_val, \
           R_test, X_dtest, y_test, \
           Xpca_train, Xpca_val, Xpca_test, \
           pca_train, pca_val, pca_test
           

def load_raw_train_and_test_set_split_by_AR(path, ts = 0.2,
                              seed_no = 0, threshold = -5, 
                              ubr = None):
    data = pd.read_csv(path)
    
    # split train and test data under each Active Region
    D = data.shape[1] - 2
    
    X = data.iloc[:,:D].to_numpy()
    y = data.iloc[:,D].to_numpy().astype('float')
    

    X = np.delete(X, (5, 11, 16, 23), axis = 1)
    D -= 8
    
    if ubr is not None:
        X, y = reblance_data(X, y, threshold, ubr)
    
    
    R, X = X[:,0:4],  X[:,4:]
    
    
    R_train, X_train, y_train = np.zeros((1, 4)), np.zeros((1, D)), np.zeros(1)
    R_test, X_test, y_test = np.zeros((1,4)), np.zeros((1, D)), np.zeros(1)
    
    for l in np.sort(np.unique(R[:,0])):
        idl = (R[:,0] == l)
        

        Rl, Xl, yl = R[idl,:], X[idl,:], y[idl]
            
        
        if Rl.shape[0] < 4:
            R_train = np.concatenate((R_train, Rl))
            y_train = np.concatenate((y_train, yl))
            X_train = np.vstack((X_train, Xl))
        else:
            indices = np.arange(Rl.shape[0])
            Xtr, Xts, ytr, yts, idtr, idts = train_test_split(Xl, yl, indices,  
                                                              test_size=ts, 
                                                              random_state=seed_no)
            
            R_train = np.concatenate((R_train, Rl[idtr,:]))
            y_train = np.concatenate((y_train, ytr))
            X_train = np.vstack((X_train, Xtr))
            
            R_test = np.concatenate((R_test, Rl[idts,:]))
            y_test = np.concatenate((y_test, yts))
            X_test = np.vstack((X_test, Xts))
    
    # remove the 1st dummy entry
    R_train, X_train, y_train= R_train[1:,], X_train[1:,], y_train[1:,]
    R_test, X_test, y_test = R_test[1:,], X_test[1:,], y_test[1:,]
    
    
    
    # standardize each column of train and test separately
    X_test = StandardScaler().fit_transform(X_test)
    
    # create pca projection
    pca_test= PCA(n_components=7).fit(X_test)
    Xpca_test = pca_test.transform(X_test)
    
    # Create design matrix    
    X_dtest = np.ones( (X_test.shape[0], X_test.shape[1] + 1))
    X_dtest[:,1:] = X_test  
    
    return np.hstack((R_train, X_train)), y_train, \
           R_test, X_dtest, y_test, Xpca_test, pca_test
           

def load_raw_train_and_test_set_split_by_AR_dropsingle(path, ts = 0.2,
                              seed_no = 0, threshold = -5, 
                              ubr = None):
    data = pd.read_csv(path)
    
    # split train and test data under each Active Region
    D = data.shape[1] - 2
    
    X = data.iloc[:,:D].to_numpy()
    y = data.iloc[:,D].to_numpy().astype('float')
    

    X = np.delete(X, (5, 11, 16, 23), axis = 1)
    D -= 8    
    
    R, X = X[:,0:4],  X[:,4:]
    
    
    R_train, X_train, y_train = np.zeros((1, 4)), np.zeros((1, D)), np.zeros(1)
    R_test, X_test, y_test = np.zeros((1,4)), np.zeros((1, D)), np.zeros(1)
    
    for l in np.sort(np.unique(R[:,0])):
        idl = (R[:,0] == l)
        

        Rl, Xl, yl = R[idl,:], X[idl,:], y[idl]
            
        
        if Rl.shape[0] > 1:
            indices = np.arange(Rl.shape[0])
            Xtr, Xts, ytr, yts, idtr, idts = train_test_split(Xl, yl, indices,  
                                                              test_size=ts, 
                                                              random_state=seed_no)
            
            R_train = np.concatenate((R_train, Rl[idtr,:]))
            y_train = np.concatenate((y_train, ytr))
            X_train = np.vstack((X_train, Xtr))
            
            R_test = np.concatenate((R_test, Rl[idts,:]))
            y_test = np.concatenate((y_test, yts))
            X_test = np.vstack((X_test, Xts))
    
    # remove the 1st dummy entry
    R_train, X_train, y_train= R_train[1:,], X_train[1:,], y_train[1:,]
    R_test, X_test, y_test = R_test[1:,], X_test[1:,], y_test[1:,]
        
    # reblance the train data
    if ubr is not None:
        R_train, X_train, y_train = reblance_data2(R_train, X_train, y_train, 
                                                   threshold, ubr)
    
    
    # standardize each column of train and test separately
    train_mean = X_train.mean(axis = 0, keepdims = 1)
    train_sd = X_train.std(axis = 0, keepdims = 1)
    
    X_train = (X_train - train_mean) / train_sd
    X_test =(X_test - train_mean) / train_sd
    
    # create pca projection
    pca_test= PCA(n_components=7).fit(X_test)
    Xpca_test = pca_test.transform(X_test)
    
    # Create design matrix    
    X_dtest = np.ones( (X_test.shape[0], X_test.shape[1] + 1))
    X_dtest[:,1:] = X_test  
    
    return np.hstack((R_train, X_train)), y_train, \
           R_test, X_dtest, y_test, Xpca_test, pca_test
           
           
def load_raw_train_and_test_set_split_by_AR_coinflip(path, ts = 0.2,
                              seed_no = 0, threshold = -5, 
                              ubr = None):
    data = pd.read_csv(path)
    
    # split train and test data under each Active Region
    D = data.shape[1] - 2
    
    X = data.iloc[:,:D].to_numpy()
    y = data.iloc[:,D].to_numpy().astype('float')
    

    X = np.delete(X, (5, 11, 16, 23), axis = 1)
    D -= 8
        
    
    R, X = X[:,0:4],  X[:,4:]
    
    
    R_train, X_train, y_train = np.zeros((1, 4)), np.zeros((1, D)), np.zeros(1)
    R_test, X_test, y_test = np.zeros((1,4)), np.zeros((1, D)), np.zeros(1)
    
    for l in np.sort(np.unique(R[:,0])):
        idl = (R[:,0] == l)
        

        Rl, Xl, yl = R[idl,:], X[idl,:], y[idl]
            
        
        if Rl.shape[0] > 1:
            indices = np.arange(Rl.shape[0])
            Xtr, Xts, ytr, yts, idtr, idts = train_test_split(Xl, yl, indices,  
                                                              test_size=ts, 
                                                              random_state=seed_no)
            
            R_train = np.concatenate((R_train, Rl[idtr,:]))
            y_train = np.concatenate((y_train, ytr))
            X_train = np.vstack((X_train, Xtr))
            
            R_test = np.concatenate((R_test, Rl[idts,:]))
            y_test = np.concatenate((y_test, yts))
            X_test = np.vstack((X_test, Xts))
        else:
            # flip a coin to assign to train or test
            if np.random.uniform() <= ts:
                R_test = np.concatenate((R_test, Rl))
                y_test = np.concatenate((y_test, yl))
                X_test = np.vstack((X_test, Xl))
                
            else:
                R_train = np.concatenate((R_train, Rl))
                y_train = np.concatenate((y_train, yl))
                X_train = np.vstack((X_train, Xl))
            
    
    # remove the 1st dummy entry
    R_train, X_train, y_train= R_train[1:,], X_train[1:,], y_train[1:,]
    R_test, X_test, y_test = R_test[1:,], X_test[1:,], y_test[1:,]
    
    # rebalance the train data
    if ubr is not None:
        R_train, X_train, y_train = reblance_data2(R_train, X_train, y_train, 
                                                   threshold, ubr)
    
    
    # standardize each column of train and test separately
    train_mean = X_train.mean(axis = 0, keepdims = 1)
    train_sd = X_train.std(axis = 0, keepdims = 1)
    
    X_train = (X_train - train_mean) / train_sd
    X_test =(X_test - train_mean) / train_sd
    
    # create pca projection
    pca_test= PCA(n_components=7).fit(X_test)
    Xpca_test = pca_test.transform(X_test)
    
    # Create design matrix    
    X_dtest = np.ones( (X_test.shape[0], X_test.shape[1] + 1))
    X_dtest[:,1:] = X_test  
    
    return np.hstack((R_train, X_train)), y_train, \
           R_test, X_dtest, y_test, Xpca_test, pca_test


def load_raw_train_and_test_set_split_by_AR_random(path, ts = 0.2,
                              seed_no = 0, threshold = -5, 
                              ubr = None):
    data = pd.read_csv(path)
    
    # split train and test data under each Active Region
    D = data.shape[1] - 2
    
    X = data.iloc[:,:D].to_numpy()
    y = data.iloc[:,D].to_numpy().astype('float')
    

    X = np.delete(X, (5, 11, 16, 23), axis = 1)
    D -= 8    
    
    R, X = X[:,0:4],  X[:,4:]
    
    # split train and test set
    indices =  np.arange(R.shape[0])
    X_train, X_test, y_train, y_test, idtr, idts  = train_test_split(X, y, indices, 
                                                       test_size=ts, 
                                                       random_state=seed_no)
    R_train = R[idtr,:]
    R_test = R[idts,:]
    
    if ubr is not None:
        R_train, X_train, y_train = reblance_data2(R_train, X_train, y_train, 
                                                   threshold, ubr)
    
    # standardize each column of train and test separately
    train_mean = X_train.mean(axis = 0, keepdims = 1)
    train_sd = X_train.std(axis = 0, keepdims = 1)
    
    X_train = (X_train - train_mean) / train_sd
    X_test =(X_test - train_mean) / train_sd
    
    # create pca projection
    pca_test= PCA(n_components=7).fit(X_test)
    Xpca_test = pca_test.transform(X_test)
    
    # Create design matrix    
    X_dtest = np.ones( (X_test.shape[0], X_test.shape[1] + 1))
    X_dtest[:,1:] = X_test  
    
    return np.hstack((R_train, X_train)), y_train, \
           R_test, X_dtest, y_test, Xpca_test, pca_test
           

def create_train_and_val_set_split_by_AR(X, y, vs = 0.2, seed_no = 0):
    D = X.shape[1] - 4
    
    R, X = X[:,:4], X[:,4:]
    
    R_train, X_train, y_train = np.zeros((1, 4)), np.zeros((1, D)), np.zeros(1)
    R_val, X_val, y_val = np.zeros((1,4)), np.zeros((1, D)), np.zeros(1)
    
    for l in np.sort(np.unique(R[:,0])):
        idl = (R[:,0] == l)
        
        Rl, Xl, yl = R[idl,:], X[idl,:], y[idl]
            
        if Rl.shape[0] < 4:
            R_train = np.concatenate((R_train, Rl))
            y_train = np.concatenate((y_train, yl))
            X_train = np.vstack((X_train, Xl))
        else:
            indices = np.arange(Rl.shape[0])
            Xtr, Xts, ytr, yts, idtr, idts = train_test_split(Xl, yl, indices,  
                                                              test_size=vs, 
                                                              random_state=seed_no)
            
            R_train = np.vstack((R_train, Rl[idtr,:]))
            y_train = np.concatenate((y_train, ytr))
            X_train = np.vstack((X_train, Xtr))
            
            R_val = np.vstack((R_val, Rl[idts,:]))
            y_val = np.concatenate((y_val, yts))
            X_val = np.vstack((X_val, Xts))
    
    # remove the 1st dummy entry
    R_train, X_train, y_train= R_train[1:,], X_train[1:,], y_train[1:,]
    R_val, X_val, y_val = R_val[1:,], X_val[1:,], y_val[1:,]
    
    # standardize each column of train and test separately
    X_train = StandardScaler().fit_transform(X_train)
    X_val = StandardScaler().fit_transform(X_val)
    
    # create pca projection
    pca_train = PCA(n_components=7).fit(X_train)
    Xpca_tr = pca_train.transform(X_train)
    
    pca_val= PCA(n_components=7).fit(X_val)
    Xpca_val = pca_val.transform(X_val)
    
    # Create design matrix    
    X_dtrain = np.ones( (X_train.shape[0], X_train.shape[1] + 1))
    X_dtrain[:,1:] = X_train
    X_dval = np.ones( (X_val.shape[0], X_val.shape[1] + 1))
    X_dval[:,1:] = X_val
    
    return R_train, X_dtrain, y_train, Xpca_tr, pca_train, \
           R_val, X_dval, y_val, Xpca_val, pca_val \
               
               
def create_train_and_val_set_split_by_AR_dropsingle(X, y, vs = 0.2, seed_no = 0):
    D = X.shape[1] - 4
    
    R, X = X[:,:4], X[:,4:]
    
    R_train, X_train, y_train = np.zeros((1, 4)), np.zeros((1, D)), np.zeros(1)
    R_val, X_val, y_val = np.zeros((1,4)), np.zeros((1, D)), np.zeros(1)
    
    for l in np.sort(np.unique(R[:,0])):
        idl = (R[:,0] == l)
        
        Rl, Xl, yl = R[idl,:], X[idl,:], y[idl]
            
        if Rl.shape[0] > 1:
            indices = np.arange(Rl.shape[0])
            Xtr, Xts, ytr, yts, idtr, idts = train_test_split(Xl, yl, indices,  
                                                              test_size=vs, 
                                                              random_state=seed_no)
            
            R_train = np.vstack((R_train, Rl[idtr,:]))
            y_train = np.concatenate((y_train, ytr))
            X_train = np.vstack((X_train, Xtr))
            
            R_val = np.vstack((R_val, Rl[idts,:]))
            y_val = np.concatenate((y_val, yts))
            X_val = np.vstack((X_val, Xts))
    
    # remove the 1st dummy entry
    R_train, X_train, y_train= R_train[1:,], X_train[1:,], y_train[1:,]
    R_val, X_val, y_val = R_val[1:,], X_val[1:,], y_val[1:,]
    
    
    # create pca projection
    pca_train = PCA(n_components=7).fit(X_train)
    Xpca_tr = pca_train.transform(X_train)
    
    pca_val= PCA(n_components=7).fit(X_val)
    Xpca_val = pca_val.transform(X_val)
    
    # Create design matrix    
    X_dtrain = np.ones( (X_train.shape[0], X_train.shape[1] + 1))
    X_dtrain[:,1:] = X_train
    X_dval = np.ones( (X_val.shape[0], X_val.shape[1] + 1))
    X_dval[:,1:] = X_val
    
    return R_train, X_dtrain, y_train, Xpca_tr, pca_train, \
           R_val, X_dval, y_val, Xpca_val, pca_val 

def create_train_and_val_set_split_by_AR_coinflip(X, y, vs = 0.2, seed_no = 0):
    D = X.shape[1] - 4
    
    R, X = X[:,:4], X[:,4:]
    
    R_train, X_train, y_train = np.zeros((1, 4)), np.zeros((1, D)), np.zeros(1)
    R_val, X_val, y_val = np.zeros((1,4)), np.zeros((1, D)), np.zeros(1)
    
    for l in np.sort(np.unique(R[:,0])):
        idl = (R[:,0] == l)
        
        Rl, Xl, yl = R[idl,:], X[idl,:], y[idl]
            
        if Rl.shape[0] > 1:
            indices = np.arange(Rl.shape[0])
            Xtr, Xts, ytr, yts, idtr, idts = train_test_split(Xl, yl, indices,  
                                                              test_size=vs, 
                                                              random_state=seed_no)
            
            
            R_train = np.vstack((R_train, Rl[idtr,:]))
            y_train = np.concatenate((y_train, ytr))
            X_train = np.vstack((X_train, Xtr))
            
            R_val = np.vstack((R_val, Rl[idts,:]))
            y_val = np.concatenate((y_val, yts))
            X_val = np.vstack((X_val, Xts))
        else:
            R_train = np.concatenate((R_train, Rl))
            y_train = np.concatenate((y_train, yl))
            X_train = np.vstack((X_train, Xl))
    
    # remove the 1st dummy entry
    R_train, X_train, y_train= R_train[1:,], X_train[1:,], y_train[1:,]
    R_val, X_val, y_val = R_val[1:,], X_val[1:,], y_val[1:,]
    
    
    # create pca projection
    pca_train = PCA(n_components=7).fit(X_train)
    Xpca_tr = pca_train.transform(X_train)
    
    pca_val= PCA(n_components=7).fit(X_val)
    Xpca_val = pca_val.transform(X_val)
    
    # Create design matrix    
    X_dtrain = np.ones( (X_train.shape[0], X_train.shape[1] + 1))
    X_dtrain[:,1:] = X_train
    X_dval = np.ones( (X_val.shape[0], X_val.shape[1] + 1))
    X_dval[:,1:] = X_val
    
    return R_train, X_dtrain, y_train, Xpca_tr, pca_train, \
           R_val, X_dval, y_val, Xpca_val, pca_val 
    

def create_train_and_val_set_split_by_AR_random(X, y, vs = 0.2, seed_no = 0):    
    R, X = X[:,:4], X[:,4:]

    
    # split train and val sets
    indices = np.arange(R.shape[0])
    X_train, X_val, y_train, y_val, idtr, idts = train_test_split(X, y, indices,  
                                                      test_size=vs, 
                                                      random_state=seed_no)
    R_train = R[idtr,:]
    R_val =  R[idts,:]
       
    # create pca projection
    pca_train = PCA(n_components=7).fit(X_train)
    Xpca_tr = pca_train.transform(X_train)
    
    pca_val= PCA(n_components=7).fit(X_val)
    Xpca_val = pca_val.transform(X_val)
    
    # Create design matrix    
    X_dtrain = np.ones( (X_train.shape[0], X_train.shape[1] + 1))
    X_dtrain[:,1:] = X_train
    X_dval = np.ones( (X_val.shape[0], X_val.shape[1] + 1))
    X_dval[:,1:] = X_val
    
    return R_train, X_dtrain, y_train, Xpca_tr, pca_train, \
           R_val, X_dval, y_val, Xpca_val, pca_val 



# metrics ultilities
def compute_high_flare_rmse(y_true, y_pred, threshold = -5):
    idx = (y_true >= threshold)
    yt_true, yt_pred = y_true[idx], y_pred[idx]
    
    return np.sqrt(np.mean(np.square(yt_true - yt_pred)))

def compute_low_flare_rmse(y_true, y_pred, threshold = -5):
    idx = (y_true < threshold)
    yt_true, yt_pred = y_true[idx], y_pred[idx]
    
    return np.sqrt(np.mean(np.square(yt_true - yt_pred)))
    
    
def compute_sbd_rmse(y_true, y_pred):
    xrmse = mrmse = crmse = brmse = 0
    
    x_idx = (y_true >= -4)
    xy_true, xy_pred = y_true[x_idx], y_pred[x_idx]
    xrmse = np.sqrt(np.mean(np.square(xy_true - xy_pred)))
    
    m_idx = (y_true >= -5) & (y_true < -4)
    my_true, my_pred = y_true[m_idx], y_pred[m_idx]
    mrmse = np.sqrt(np.mean(np.square(my_true - my_pred)))
    
    c_idx = (y_true >= -6) & (y_true < -5)
    cy_true, cy_pred = y_true[c_idx], y_pred[c_idx]
    crmse = np.sqrt(np.mean(np.square(cy_true - cy_pred)))
    
    b_idx = (y_true < -6)
    by_true, by_pred = y_true[b_idx], y_pred[b_idx]
    brmse = np.sqrt(np.mean(np.square(by_true - by_pred)))
     
    return xrmse, mrmse, crmse, brmse

def compute_classification_metrics(z_true, z_pred):
    tn, fp, fn, tp = 1.0 * confusion_matrix(z_true, z_pred).ravel()
    print(tn, fp, fn, tp)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    hss1 = recall * (2 - 1/precision)
    hss2 = 2 * (tp * tn - fp * fn) / ((tp + fn)*(fn + tn) + (tp + fp)*(fp + tn))
    tss = tp /(tp + fn) - fp / (fp + tn)
    acc = np.sum(z_true == z_pred) * 1.0 / len(z_true)
    
    return precision, recall, f1, hss1, hss2, tss, acc

def compute_performance_metrics(y_true, y_pred, clf_threshold = -5):
    rmse = np.sqrt(np.mean(np.square(y_true - y_pred)))
    
    z_true = convert_to_flare_bin_label(y_true, -5)
    z_pred = convert_to_flare_bin_label(y_pred, clf_threshold)
    
    precision, recall, f1, hss1, hss2, tss, acc = compute_classification_metrics(z_true, z_pred)
    hrmse = compute_high_flare_rmse(y_true, y_pred)
    lrmse = compute_low_flare_rmse(y_true, y_pred)
    xrmse, mrsme, crmse, brmse = compute_sbd_rmse(y_true, y_pred)
    
    
    return rmse, precision, recall, f1, hss1, hss2, tss, acc, hrmse, lrmse, xrmse, mrsme, crmse, brmse


def print_metric(metrics):
    fmt_str = "Model performance on train data: rmse = {}, precision = {}," + \
              " recall = {}, f1 = {}, hss1 = {}, hss2 = {}, tss = {}, acc = {}," + \
              " hi_rmse = {}, lo_rmse = {}, xrmse = {}, mrmse = {}, crmse = {}, brmse = {}"
    
    print(fmt_str.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4],
                         metrics[5], metrics[6], metrics[7], metrics[8], metrics[9],
                         metrics[10], metrics[11], metrics[12], metrics[13]))


def print_model_performance(model, X_train, y_train, pcatr, X_test, y_test, pcats):
    y_tr_pred = model.predict(X_train)
    tr_metrics = compute_performance_metrics(y_train, y_tr_pred)
    print_metric(tr_metrics)
    plot_binary_contour(y_tr_pred, pcatr)
    
    
    y_ts_pred = model.predict(X_test)
    ts_metrics = compute_performance_metrics(y_test, y_ts_pred)
    print_metric(ts_metrics)
    plot_binary_contour(y_ts_pred, pcats)
    
    return tr_metrics, ts_metrics



def print_amodel_performance(model, X_train, y_train, pcatr, threshold = -5):
    y_tr_pred = model.predict(X_train)
    tr_metrics = compute_performance_metrics(y_train, y_tr_pred, clf_threshold = threshold)
    print_metric(tr_metrics)
    plot_binary_contour(y_tr_pred, pcatr)
    
    return tr_metrics

def get_amodel_performance(model, X_train, y_train):
    y_tr_pred = model.predict(X_train)
    tr_metrics = compute_performance_metrics(y_train, y_tr_pred)
    
    return tr_metrics


def print_mm_performance(model, R, X_train, y_train, pcatr, R_test, X_test, y_test, pcats):
    y_tr_pred = model.predict_y(R, X_train)
    tr_metrics = compute_performance_metrics(y_train, y_tr_pred)
    print_metric(tr_metrics)
    plot_binary_contour(y_tr_pred, pcatr)
    
    
    y_ts_pred = model.predict_y(R_test, X_test)
    ts_metrics = compute_performance_metrics(y_test, y_ts_pred)
    print_metric(ts_metrics)
    plot_binary_contour(y_ts_pred, pcats)
    
    return tr_metrics, ts_metrics


def print_amm_performance(model, R, X_train, y_train, pcatr, threshold = -5):
    y_tr_pred = model.predict_y(R, X_train)
    tr_metrics = compute_performance_metrics(y_train, y_tr_pred, clf_threshold = threshold)
    print_metric(tr_metrics)
    plot_binary_contour(y_tr_pred, pcatr)
    
    return tr_metrics

def get_amm_performance(model, R, X_train, y_train):
    y_tr_pred = model.predict_y(R, X_train)
    tr_metrics = compute_performance_metrics(y_train, y_tr_pred)
    
    return tr_metrics


def get_model_predict_values(model, R, X_train):
    return model.predict_y(R, X_train).copy()


# plot ultilities
def plot_y(y_true, y_pred, nsize = 200):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    
    ax[0].scatter(y_true, y_pred, color = 'red')
    ax[0].set_xlabel("y_true")
    ax[0].set_ylabel("y_pred")
    ax[0].set_xlim((-7.5, -3))
    ax[0].set_ylim((-7.5, -3))
    
    ax[1].plot(range(nsize), y_true[:nsize], color = 'gray')  
    ax[1].plot(range(nsize), y_pred[:nsize], color= 'red')  
    ax[1].set_xlabel("Point #")
    ax[1].set_ylabel("y")
    
    fig.tight_layout()
    
def plot_cluster_assignments(r, pcax, y, mu = None, transformer = None, z = None):
    markers = ['*', 'v', '.','x','o']
    
    def plot_y_group(i, j, m):            
        for k in np.unique(z):
            ax[i,j].scatter(pcax[z==k, m], y[z==k] , color= colors[k], s=15, 
              marker = markers[k])
    
    def plot_X_group(i, j, m, n):            
        for k in np.unique(z):
            ax[i,j].scatter(pcax[z==k, m], pcax[z==k, n],  color= colors[k], s=15, 
              marker = markers[k])
            
        if mu is not None:
            ax[i,j].scatter(mu_pca[:, m], mu_pca[:, n], color = 'red')
            
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4))
    
    if z is None:
        z = r.argmax(axis=1)
    
    if mu is not None and transformer is not None:
        mu_pca = transformer.transform(mu[:,1:])
    
    
    plot_y_group(0, 0, 0)
    ax[0,0].set_xlabel('PCA1')
    ax[0,0].set_ylabel('y')
    
    plot_y_group(0, 1, 1)
    ax[0,1].set_xlabel('PCA2')
    ax[0,1].set_ylabel('y')
    
    plot_y_group(0, 2, 2)
    ax[0,2].set_xlabel('PCA3')
    ax[0,2].set_ylabel('y')
    
    plot_X_group(1, 0, 0, 1)
    ax[1,0].set_xlabel('PCA1')
    ax[1,0].set_ylabel('PCA2')
    
    plot_X_group(1, 1, 0, 2)
    ax[1,1].set_xlabel('PCA1')
    ax[1,1].set_ylabel('PCA3')
    
    plot_X_group(1, 2, 1, 2)
    ax[1,2].set_xlabel('PCA2')
    ax[1,2].set_ylabel('PCA3')
    
    fig.tight_layout()
    

def plot_cluster_assignments_by_aslide(r, pcax, y, dim1 = 0, dim2 = 1):
    markers = ['*', 'v', '.','x','o']
    
    def plot_y_group(i, m):   
        K = len(np.unique(z))         
        for k in np.sort(np.unique(z)):
            ax[i,k].scatter(pcax[z==k, m], y[z==k] , color= colors[k], s=15, 
              marker = markers[k])
            ax[i,k].set_xlabel('PCA' + str(m))
            ax[i,k].set_ylabel('y')
            ax[i,k].axhline(y = -5)
            ax[i,k].set_ylim((-7.55, -3))
            ax[i,K].scatter(pcax[z==k, m], y[z==k] , color= colors[k], s=15, 
              marker = markers[k])
            ax[i,K].set_xlabel('PCA' + str(m))
            ax[i,K].set_ylabel('y')
            ax[i,K].set_ylim((-7.55, -3))
            ax[i,K].axhline(y = -5)
    
    def plot_X_group(i, m, n):
        K = len(np.unique(z))               
        for k in np.sort(np.unique(z)):
            ax[i, k].scatter(pcax[z==k, m], pcax[z==k, n],  color= colors[k], s=15, 
              marker = markers[k])
            ax[i,k].set_xlabel('PCA' + str(m))
            ax[i,k].set_ylabel('PCA' + str(n))
            ax[i, K].scatter(pcax[z==k, m], pcax[z==k, n],  color= colors[k], s=15, 
              marker = markers[k])
            ax[i,K].set_xlabel('PCA' + str(m))
            ax[i,K].set_ylabel('PCA' + str(n))
    
    z = r.argmax(axis=1)
    K = len(np.unique(z))
    fig, ax = plt.subplots(nrows=3, ncols=K+1, figsize=(8, 4))
        
    plot_y_group(0, 0)

    
    plot_y_group(0, dim1)
    plot_y_group(1, dim2)
    plot_X_group(2, dim1, dim2)
    
    fig.tight_layout()
        
    
def plot_beta_hat(beta_hat):
    colors = ['black', 'crimson', 'violet']
    
    K = beta_hat.shape[0]
    fig, ax = plt.subplots(nrows=1, ncols=K, figsize=(8, 4))
    
    ymin, ymax = np.min(beta_hat), np.max(beta_hat)
    
    for k in range(K):
        feat_nums = range(1, len(beta_hat[k,:]) + 1)
        ax[k].bar(feat_nums, beta_hat[k,:], align='center', 
                  color = colors[k])
        ax[k].set_xticks(feat_nums, feat_nums)
        ax[k].set_ylabel("coefficient beta " + str(k + 1))
        ax[k].set_ylim((ymin, ymax))
    
    fig.tight_layout()
    
def plot_mu_hat(mu_hat):
    K = mu_hat.shape[0]
    fig, ax = plt.subplots(nrows=1, ncols=K, figsize=(8, 4))
    
    
    for k in range(K):
        feat_nums = range(len(mu_hat[k,:]))
        ax[k].bar(feat_nums, mu_hat[k,:], align='center', alpha=0.5)
        ax[k].set_xticks(feat_nums, feat_nums)
        ax[k].set_ylabel("mu " + str(k))
    
    fig.tight_layout()    
    
def plot_Sigmad_hat(Sigmad_hat):
    K = Sigmad_hat.shape[0]
    fig, ax = plt.subplots(nrows=1, ncols=K, figsize=(8, 4))
    
    
    for k in range(K):
        feat_nums = range(len(Sigmad_hat[k,:]))
        ax[k].bar(feat_nums, Sigmad_hat[k,:], align='center', alpha=0.5)
        ax[k].set_xticks(feat_nums, feat_nums)
        ax[k].set_ylabel("Sigma " + str(k))
    
    fig.tight_layout()
    
    
def plot_gamma_hat(gamma_hat):
    feat_nums = range(len(gamma_hat[0,:]))
    plt.bar(feat_nums, gamma_hat[0,:], align='center', alpha=0.5)
    plt.xticks(feat_nums, feat_nums)
    plt.ylabel("gamma " + str(0))
    
    plt.show()    
    
def plot_pi_hat(pi_hat):    
    cl_nums = range(len(pi_hat))
    
    for k in range(len(cl_nums)):
        plt.bar(cl_nums[k], pi_hat[k],
                color = colors[k],
                align='center')
    plt.xticks(cl_nums, np.arange(1, len(cl_nums) + 1))
    plt.ylabel('Mixing Weight')
    plt.xlabel('Cluster')
    plt.show()
    
def plot_region_pi(pir):
    K = pir.shape[1]
    fig, ax = plt.subplots(nrows=K, ncols=1, figsize=(8, 4))
    
    for k in range(K):
        regions = range(len(pir[:,k]))
    
        ax[k].bar(regions, pir[:,k], align='center', alpha=0.5, color = colors[k])
        ax[k].set_xticks(regions, regions)
        ax[k].set_ylabel('Cat ' + str(k))
        ax[k].set_ylim((0,1))
    
    fig.tight_layout()

def plot_sigma2_hat(sigma2_hat):
    cl_nums = range(len(sigma2_hat))
    
    for k in range(len(cl_nums)):
        print(colors[k])
        plt.bar(cl_nums[k], sigma2_hat[k],
                color = colors[k],
                align='center')
        
    plt.xticks(cl_nums, np.arange(1, len(cl_nums) + 1))
    plt.ylabel('sigma2')
    plt.xlabel('Cluster')
    plt.show()
    
def plot_betais(X, y, r, beta_hat):    
    betais = np.zeros((X.shape[0], X.shape[1]))
    
    for n in range(X.shape[0]):
        betais[n, ] = np.linalg.pinv(X[n,].reshape(X.shape[1],1)) * y[n]
        
        
    pca = PCA(n_components=3).fit(betais)
    pbetais = pca.transform(betais)
    pbehat =  pca.transform(beta_hat)
    
    plt.scatter(pbetais[:,0], pbetais[:,1])
    plt.scatter(pbehat[0,0], pbehat[0,1], color = 'red')
    plt.scatter(pbehat[1,0], pbehat[1,1], color = 'blue')
    plt.show()
    
    
def plot_betai(betas, mu, r, betalr):
    markers = ['*', 'v']
    
    def plot_group(i):
        ax[i].scatter(mux[:, 0], mux[:, 1], color = 'red')
        ax[i].scatter(blrx[:, 0], blrx[:, 1], color = 'black')
            
        for k in np.unique(z):
            ax[i].scatter(betax[z == k, 0], betax[z == k, 1] , color= colors[k], s=15, 
              marker = markers[k])
            
    pca = PCA(n_components=3).fit(betas)
    
    betax = pca.transform(betas)
    mux = pca.transform(mu)
    blrx = pca.transform(betalr)
    
    z = r.argmax(axis=1)    
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    
    plot_group(0)
    ax[0].set_xlabel('pca beta1')
    ax[0].set_ylabel('pca beta2')
    
    plot_group(1)
    ax[1].set_xlabel('pca beta1')
    ax[1].set_ylabel('pca beta3')
    
    plot_group(2)
    ax[2].set_xlabel('pca beta2')
    ax[2].set_ylabel('pca beta3')
    
    fig.tight_layout()

def plot_cluster_betas(betas, sizer, mu, r, betalr = None, betamm0 = None,
                       recoord = False):        
    pca = PCA(n_components=3).fit(betas)
    
    betax = pca.transform(betas)
    
    mux = pca.transform(mu)
    
    if recoord:
        print("Recomputing...")
        for n in range(betax.shape[0]):
            betax[n, ] = np.zeros(betax.shape[1])
            
            for k in range(r.shape[1]):
                betax[n, ] += r[n,k] * mux[k,]
                
    betamm0x, blrx = None, None
    
    if betamm0 is not None:
        betamm0x = pca.transform(betamm0)
    if betalr is not None:
        blrx = pca.transform(betalr)
    
    z = r.argmax(axis=1)
    
    def plot_slide(i, m, n):        
        for k in np.sort(np.unique(z)):
            ax[i].scatter(mux[k, m], mux[k, n], color = 'lightgray',
                          s = 10 * np.max(sizer[z==k]), marker = markes[k])
            ax[i].scatter(betax[z==k, m], betax[z==k, n] ,color = colors[k], 
                          s= 3 * sizer[z==k], marker = markes[k])
        
        if blrx is not None:
            ax[i].scatter(blrx[:, m], blrx[:, n], color = 'red')
        if betamm0x is not None:
            ax[i].scatter(betamm0x[:, m], betamm0x[:, n], color = 'black')
            
        ax[i].set_xlabel('pca beta' + str(m + 1))
        ax[i].set_ylabel('pca beta' + str(n + 1))
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    plot_slide(0, 0, 1)
    plot_slide(1, 0, 2)
    plot_slide(2, 1, 2)
    fig.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for k in np.sort(np.unique(z)):
        ax.scatter(mux[k, 0], mux[k, 1], mux[k, 2], color = 'lightgray', marker= markes[k],
                   s = 10 * np.max(sizer[z==k]))
        
        ax.scatter(betax[z==k, 0], betax[z==k, 1], betax[z==k, 2], marker= markes[k], 
                   color = colors[k], s = 3 * sizer[z==k])
    
    if betamm0x is not None:
        ax.scatter(betamm0x[:, 0], betamm0x[:, 1], betamm0x[:, 2], color = 'black', s = 10)
    
    if blrx is not None:
        ax.scatter(blrx[:, 0], blrx[:, 1], blrx[:, 2], color = 'red', s = 10)
        
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    
    fig.tight_layout()
    
    
def plot_two_betaclusters(beta1s, beta1_cen, r1, sizer1,
                          beta2s, beta2_cen, r2, sizer2, 
                          recoord = False):        
    betas = np.vstack((beta1s, beta2s))
    
    pca = PCA(n_components=3).fit(betas)
    
    
    beta1x = pca.transform(beta1s)
    mu1x = pca.transform(beta1_cen)
    
    beta2x = pca.transform(beta2s)
    mu2x = pca.transform(beta2_cen)
    
    if recoord:
        print("Recomputing...")
        for n in range(beta1x.shape[0]):
            beta1x[n, ] = np.zeros(beta1x.shape[1])
            
            for k in range(r1.shape[1]):
                beta1x[n, ] += r1[n,k] * mu1x[k,]
                
        for n in range(beta2x.shape[0]):
            beta2x[n, ] = np.zeros(beta2x.shape[1])
            
            for k in range(r2.shape[1]):
                beta2x[n, ] += r2[n,k] * mu2x[k,]
                
    z1 = r1.argmax(axis=1)
    z2 = r2.argmax(axis=1)
    
    def plot_slide(i, m, n, z, betax, mux, sizer, colors, gradient = 1.0):        
        for k in np.sort(np.unique(z)):
            ax[i].scatter(mux[k, m], mux[k, n], color = colors[i], alpha = gradient,
                          s = 10 * np.max(sizer[z==k]), marker = markes[k])
            ax[i].scatter(betax[z==k, m], betax[z==k, n] ,color = colors[k], 
                          s= 3 * sizer[z==k], marker = markes[k], 
                          alpha = gradient)
            
        ax[i].set_xlabel('pca beta' + str(m + 1))
        ax[i].set_ylabel('pca beta' + str(n + 1))
    
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    
    alt_colors = ['dimgray', 'dimgrey','gray','grey','darkgray','darkgrey',
                  'silver']
    
    # plot first set of betas
    plot_slide(0, 0, 1, z1, beta1x, mu1x, sizer1, colors)
    plot_slide(1, 0, 2, z1, beta1x, mu1x, sizer1, colors)
    plot_slide(2, 1, 2, z1, beta1x, mu1x, sizer1, colors)
    
    # plot 2nd set of betas
    plot_slide(0, 0, 1, z2, beta2x, mu2x, sizer2, alt_colors, gradient = 0.6)
    plot_slide(1, 0, 2, z2, beta2x, mu2x, sizer2, alt_colors, gradient = 0.6)
    plot_slide(2, 1, 2, z2, beta2x, mu2x, sizer2, alt_colors, gradient = 0.6)
    
    fig.tight_layout()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for k in np.sort(np.unique(z1)):
        ax.scatter(mu1x[k, 0], mu1x[k, 1], mu1x[k, 2], color = colors[k], marker= markes[k],
                   s = 10 * np.max(sizer1[z1==k]))
        
        ax.scatter(beta1x[z1==k, 0], beta1x[z1==k, 1], beta1x[z1==k, 2], marker= markes[k], 
                   color = colors[k], s = 3 * sizer1[z1==k])
        
    for k in np.sort(np.unique(z2)):
        ax.scatter(mu2x[k, 0], mu2x[k, 1], mu2x[k, 2], color = alt_colors[k], marker= markes[k],
                   s = 10 * np.max(sizer2[z2==k]), alpha = 0.6)
        
        ax.scatter(beta2x[z2==k, 0], beta2x[z2==k, 1], beta2x[z2==k, 2], marker= markes[k], 
                   color = alt_colors[k], s = 3 * sizer2[z2==k], alpha = 0.6)
        
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')
    
    fig.tight_layout()


def plot_contour_betai(betas, mu, y, r, threshold = -5):    
    cc = ["black", "gray"]
    
    pca = PCA(n_components=3).fit(betas)
    
    betax = pca.transform(betas)
    
    mux = pca.transform(mu)   
    y_tilde = (y >= threshold).astype(int)
    z = r.argmax(axis=1)
    
    def plot_slide(row, m, n):
        I = np.sort(np.unique(y_tilde))
        for i in I:            
            ax[row, i].scatter(betax[y_tilde == i, m], betax[y_tilde == i, n], 
                               color = cc[i])
            
            ax[row, i].set_xlabel('pca beta' + str(m))
            ax[row, i].set_ylabel('pca beta' + str(n))
            
            for k in np.sort(np.unique(z)):
                ax[row,i].scatter(mux[k, m], mux[k, n], color = colors[k])
        
        for i in I:
            ax[row, len(I)].scatter(betax[y_tilde == i, m], betax[y_tilde == i, n], 
                                        color = cc[i])
    
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(8, 4))
    plot_slide(0, 0, 1)
    plot_slide(1, 0, 2)
    plot_slide(2, 1, 2)
    fig.tight_layout()
    
    fig = plt.figure()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k in np.sort(np.unique(z)):
        ax.scatter(mux[k, 0], mux[k, 1], mux[k, 2], color = colors[k], s = 200)
    
    
    for i in np.sort(np.unique(y_tilde)):            
        ax.scatter(betax[y_tilde == i, 0], betax[y_tilde == i, 1], betax[y_tilde == i, 2],
                   color = cc[i])
        
    fig.tight_layout()
    

    
def plot_test_cluster_assignments(r, pcax, y, mu = None, transformer = None, z = None):
    colors = ["green", "orange"]
    markers = ['*', 'v']
    
    def plot_y_group(i, m):            
        for k in np.unique(z):
            ax[i].scatter(pcax[z==k, m], y[z==k] , color= colors[k], s=15, 
              marker = markers[k])
    
    def plot_X_group(i, m, n):            
        for k in np.unique(z):
            ax[i].scatter(pcax[z==k, m], pcax[z==k, n],  color= colors[k], s=15, 
              marker = markers[k])
            
        if mu is not None:
            ax[i].scatter(mu_pca[:, m], mu_pca[:, n], color = 'red')
            
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 4))
    
    if z is None:
        z = r.argmax(axis=1)
    
    if mu is not None and transformer is not None:
        mu_pca = transformer.transform(mu[:,1:])
    
    
    plot_y_group(0, 0)
    ax[0].set_xlabel('PCA1')
    ax[0].set_ylabel('y')
    
    plot_y_group(1, 1)
    ax[1].set_xlabel('PCA2')
    ax[1].set_ylabel('y')
    
    plot_X_group(2, 0, 1)
    ax[2].set_xlabel('PCA1')
    ax[2].set_ylabel('PCA2')
    
    
    fig.tight_layout()
    
    
def plot_binary_contour(y, pcax, level = 2, z = None, threshold = -5):
    colors = ["black", "gray"]
    markers = ['*', 'v']
    
    def plot_X_group(i, m, n):         
        ax[i,0].scatter(pcax[z==0, m], pcax[z==0, n],  color= colors[0], s=15, 
              marker = markers[0])
        ax[i,0].set_xlabel('PCA' + str(m))
        ax[i,0].set_ylabel('PCA' + str(n))
        
        ax[i,1].scatter(pcax[z==1, m], pcax[z==1, n],  color= colors[1], s=15, 
              marker = markers[1])
        ax[i,1].set_xlabel('PCA' + str(m))
        ax[i,1].set_ylabel('PCA' + str(n))
        
        ax[i,2].scatter(pcax[z==0, m], pcax[z==0, n],  color= colors[0], s=15, 
               marker = markers[0])        
        ax[i,2].scatter(pcax[z==1, m], pcax[z==1, n],  color= colors[1], s=15, 
               marker = markers[1])
        ax[i,2].set_xlabel('PCA' + str(m))
        ax[i,2].set_ylabel('PCA' + str(n))
        ax[i,2].set_xlabel('PCA' + str(m))
        ax[i,2].set_ylabel('PCA' + str(n))
                        
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(8, 4))
    
    z = (y >= threshold).astype(int)
    
    plot_X_group(0, 0, 1)    
    plot_X_group(1, 0, 2)
    plot_X_group(2, 1, 2)      

    fig.tight_layout()

def plot_ar_cluster_assignment(r):    
    z = r.argmax(axis = 1)
    regs = np.arange(len(z))
    
    for i in np.unique(z):
        print(i)
        idx = (z == i)
        plt.bar(regs[idx], (z[idx] + 1), 
                color = colors[i],
                width = 1.0)
    
    plt.show()
    
    
def plot_loc(z, loc, major_k = [1]):    
    for k in np.unique(z):
        idk = (z == k)
        if k not in major_k:
            plt.scatter(loc[idk,0], loc[idk,1], color = colors[k], s = 10)
        else:
            plt.scatter(loc[idk,0], loc[idk,1], color = colors[k], s = 10,
                    alpha = 0.2)
        
    plt.xlabel('MDS dim 1')
    plt.ylabel('MDS dim 2')
    plt.show()
    
def extract_region_beta(mm):
    betar = np.zeros((mm.Nr, mm.D))
    size = np.full(mm.Nr, 0)
    
    for r in range(mm.Nr):
        ar_no = mm.uR[r]
        size[r] = mm.X[mm.R == ar_no, ].shape[0]
        for k in range(mm.K):
            betar[r,] += mm.tau[r,k] * mm.beta[k,]
    
    return betar, size

def extract_betai(mm):
    betai = np.zeros((mm.N, mm.D))

    for i in range(mm.N):
        for k in range(mm.K):
            betai[i,] += mm.taui[i,k] * mm.beta[k,]
    
    return betai


# Linear Regression Model
class LinearRegression:
    def __init__(self, X, y, W = None):
        self.X = X
        self.N = X.shape[0]
        self.D = X.shape[1]
        
        if W is None:
            W = np.identity(self.N)
        
        self.beta_hat = np.linalg.inv(X.T.dot(W).dot(X)).dot(X.T).dot(W).dot(y)
    
    def predict(self, X):
        return  X.dot(self.beta_hat)
    
    def predict_y(self, R, X):
        return  X.dot(self.beta_hat)


# Misc ultitilites
def convert_to_flare_bin_label(y, threshold = -5):
    return (y >= threshold)

def convert_design_matrix(X):
    Xd = np.ones((X.shape[0], X.shape[1] + 1))
    Xd[:,1:] = X
    
    return Xd


# Create Class Imblance Weighted Matrix W
def create_W_by_size(y):
    W = np.zeros((len(y), len(y)))
    
    size = [np.sum(y >= - 4), np.sum(y >= - 5), np.sum(y >= - 6), np.sum(y < -6)]
    
    for i in range(len(y)):
        W[i,i] = 1.0
     
        if y[i] >= - 4:
            W[i,i] /= size[0]
        elif y[i] >= -5:
            W[i,i] /= size[1]
        elif y[i] >= -6:
            W[i,i] /= size[2]
        else:
            W[i,i] /= size[3]
            
    return W

def create_W_by_fit_p(p, y):    
    W = np.zeros((len(y), len(y)))
    
    l, c = p.fit(y)
    fp = p(loc = l , scale = c)
    
    for i in range(len(y)):
        W[i,i] = 1.0
     
        if y[i] >= - 4:
            W[i,i] /= (1 - fp.cdf(-4))
        elif y[i] >= -5:
            W[i,i] /= (fp.cdf(-4) - fp.cdf(-5))
        elif y[i] >= -6:
            W[i,i] /= (fp.cdf(-5) - fp.cdf(-6))
        else:
            W[i,i] /= (fp.cdf(-6))
            
    return W

def create_W_by_fit_st(nu, y):    
    W = np.zeros((len(y), len(y)))
    
    _, l, s = t.fit(y, fdf = nu)
    fp = t(df = nu, loc = l , scale = s)
    
    for i in range(len(y)):
        W[i,i] = 1.0
     
        if y[i] >= - 4:
            W[i,i] /= (1 - fp.cdf(-4))
        elif y[i] >= -5:
            W[i,i] /= (fp.cdf(-4) - fp.cdf(-5))
        elif y[i] >= -6:
            W[i,i] /= (fp.cdf(-5) - fp.cdf(-6))
        else:
            W[i,i] /= (fp.cdf(-6))
            
    return W

def create_W_by_fit_pht_class(p, y):    
    W = np.zeros((len(y), len(y)))
    
    org = np.mean(y)
    
    nu, l, s = p.fit(y - org)
    fp = p(nu, loc = l , scale = s)
    
    for i in range(len(y)):
        W[i,i] = 1.0
     
        if y[i] >= - 4:
            W[i,i] /= (1 - fp.cdf(-4 - org))
        elif y[i] >= -5:
            W[i,i] /= (fp.cdf(-4 - org) - fp.cdf(-5 - org))
        elif y[i] >= -6:
            W[i,i] /= (fp.cdf(-5 - org) - fp.cdf(-6 - org))
        else:
            W[i,i] /= (fp.cdf(-6 - org))
            
    return W

# Regional statistics     
def get_region_stats(R, y):
    L = np.unique(R)
    regions = np.zeros((len(L), 7))
    
    for i, r in zip(range(len(L)), L):
        regions[i,0] = r
        idr = (R == r)
        Nr = np.sum(idr)
        
        regions[i, 1] = Nr
        regions[i, 2] = 1.0 * np.sum(y[idr] >= -5) / Nr
        regions[i, 3] = 1.0 * np.sum(y[idr] < -5) / Nr
        regions[i, 4] = 1.0 * np.percentile(y[idr], 2.5)
        regions[i, 5] = 1.0 * np.percentile(y[idr], 97.5)
        regions[i, 6] = 1.0 * np.mean(y[idr])
        
    
    return regions
    

def compute_dense_score(tau):
    score = 0
    N, K = tau.shape[0], tau.shape[1]
    
    for i in range(N):
        si = 0.0
        
        for k in range(K):
            s0 = tau[i,k]
            s1 = np.abs(tau[i,k] - 1)
            si += np.min([s0, s1]) 
            
        score += si
    
    return score

def compute_tau_reg(mm, R_test):
    R_test = np.unique(R_test.copy())
    Nt = len(R_test)
        
    tau = np.zeros((Nt, mm.K))
        
    for i, r in enumerate(R_test):
        ri = mm.ARtor[r]
        tau[i, ] = mm.tau[ri,] 
        
    return tau


def fraction_inside_simplex_center(mu, betas, alpha = 1.0):
    N = betas.shape[0]
    count = 0.0
    center = mu.mean(axis = 0)
    
    radius = alpha * np.mean(np.sqrt((np.square(mu - center)).sum(axis=0)))
    
    for i in range(N):
        ri = np.sqrt(np.sum(np.square(betas[i,] - center)))
        if ri < radius:
            count += 1.0
    
    return count / N


def get_region_size(R, X):
    uR = np.sort(np.unique(R))
    size = {}
    
    for r in uR:
        size[r] = X[R == r,].shape[0]
        
    return size


def region_and_long_lat(R):
    reg, lon, lat, tim = R[:,0],  R[:,1],  R[:,2], R[:, 3]
    
    max_thres = np.max((np.max(lon), np.max(lat)))
    tim = ((tim - tim.min()) / (tim.max() - tim.min()) - 0.5) * 2.0 * max_thres
    loc = np.column_stack((lon, lat, tim))
    
    return reg, loc

def box_plot_f1s_over_K(result_path = '../results/2R/', K = 7, delta = 6, mode = 'val'):
    mean_set = []
    fig, ax = plt.subplots()
    fig.set_size_inches(9.5, 6.5)
    box_dict = {}

    for k in range(K):
        f1_metrics = pd.read_csv(result_path + str(delta) + 'h' + mode + 
                                 '_metricsK' + str(k) + '.csv').to_numpy()[1:,3]  
        mean_set += [np.mean(f1_metrics)]
        box_dict[k] = f1_metrics

    ax.plot(np.arange(1, K + 1), mean_set, 'go--', linewidth=2, markersize=12)
    ax.set_xticklabels([k + 1 for k in box_dict.keys()])
    ax.boxplot(box_dict.values())
    ax.set_title( "F1-score")
    fig.show()
    

def create_train_perf_from_saved_models(path = 'dump', Nrep = 20, K = 7, delta = 6, outpath = 'results'):
    train_metrics2r = np.zeros((K, Nrep, 10))
    test_metrics2r = np.zeros((K, Nrep, 10))
    val_metrics2r = np.zeros((K, Nrep, 10))

    train_metrics2h = np.zeros((K, Nrep, 10))
    test_metrics2h = np.zeros((K, Nrep, 10))
    val_metrics2h = np.zeros((K, Nrep, 10))  
    
    R_test = pd.read_csv(path + '_test_R.csv').to_numpy()[:,0]
    X_test = pd.read_csv(path + '_test_X.csv').to_numpy()
    y_test = pd.read_csv(path + '_test_y.csv').to_numpy()[:,0]
    
    for rep in range(Nrep):
        R_train = pd.read_csv(path + 'r' + str(rep) + '_train_R.csv').to_numpy()[:,0]
        X_train = pd.read_csv(path + 'r' + str(rep) + '_train_X.csv').to_numpy()
        y_train = pd.read_csv(path + 'r' + str(rep) + '_train_y.csv').to_numpy()[:,0]
        
        R_val = pd.read_csv(path + 'r' + str(rep) + '_val_R.csv').to_numpy()[:,0]
        X_val = pd.read_csv(path + 'r' + str(rep) + '_val_X.csv').to_numpy()
        y_val = pd.read_csv(path + 'r' + str(rep) + '_val_y.csv').to_numpy()[:,0]
        
        
        W = create_W_by_fit_st(1, y_train)
        wlr = LinearRegression(X_train, y_train, W = W)
    
        train_metrics2r[0, rep, ] = get_amodel_performance(wlr, X_train, y_train)
        test_metrics2r[0, rep, ] = get_amodel_performance(wlr, X_test, y_test)
        val_metrics2r[0, rep, ] = get_amodel_performance(wlr, X_val, y_val)
    
        train_metrics2h[0, rep, ] = get_amodel_performance(wlr, X_train, y_train)
        test_metrics2h[0, rep, ] = get_amodel_performance(wlr, X_test, y_test)
        val_metrics2h[0, rep, ] = get_amodel_performance(wlr, X_val, y_val)
        
        for k in range(2, K + 1):
            em_run2R = pickle.load( open( path + 'mm2RK' + str(k) + 'r' + 
                                         str(rep) + '.pkl', "rb" ) )
            
            train_metrics2r[k-1, rep, ] = get_amm_performance(em_run2R['mm2R'], R_train, X_train, y_train)
            test_metrics2r[k-1, rep, ] = get_amm_performance(em_run2R['mm2R'], R_test, X_test, y_test)
            val_metrics2r[k-1, rep, ] = get_amm_performance(em_run2R['mm2R'], R_val, X_val, y_val)
            
            
            em_run2H = pickle.load( open( path + 'mm2HK' + str(k) + 'r' + 
                                         str(rep) + '.pkl', "rb" ) )
            
            train_metrics2h[k-1, rep, ] = get_amm_performance(em_run2H['mm2Ad'], R_train, X_train, y_train)
            test_metrics2h[k-1, rep, ] = get_amm_performance(em_run2H['mm2Ad'], R_test, X_test, y_test)
            val_metrics2h[k-1, rep, ] = get_amm_performance(em_run2H['mm2Ad'], R_val, X_val, y_val)
    
    for k in range(K):
        pd.DataFrame(train_metrics2r[k,]).to_csv(outpath + '/2R/' + str(delta) + 'htrain_metricsK' + str(k) + '.csv', header =False, index=False, mode = 'a')
        pd.DataFrame(val_metrics2r[k,]).to_csv(outpath + '/2R/' + str(delta) + 'hval_metricsK' + str(k) + '.csv', header =False, index=False, mode = 'a')
        pd.DataFrame(test_metrics2r[k,]).to_csv(outpath + '/2R/' + str(delta) + 'htest_metricsK' + str(k) + '.csv', header =False, index=False, mode = 'a')
    
    for k in range(K):
        pd.DataFrame(train_metrics2h[k,]).to_csv(outpath + '/2H/' + str(delta) + 'htrain_metricsK' + str(k) + '.csv', header =False, index=False, mode = 'a')
        pd.DataFrame(val_metrics2h[k,]).to_csv(outpath + '/2H/' + str(delta) + 'hval_metricsK' + str(k) + '.csv', header =False, index=False, mode = 'a')
        pd.DataFrame(test_metrics2h[k,]).to_csv(outpath + '/2H/' + str(delta) + 'htest_metricsK' + str(k) + '.csv', header =False, index=False, mode = 'a') 