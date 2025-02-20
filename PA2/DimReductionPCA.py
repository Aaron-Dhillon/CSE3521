import numpy as np
import matplotlib.pyplot as plt
import math


def PCA(X, out_dim):
    """
    Input:
        X: a D-by-N matrix (numpy array) of the input data
        out_dim: the desired output dimension
    Output:
        mu: the mean vector of X. Please represent it as a D-by-1 matrix (numpy array).
        W: the projection matrix of PCA. Please represent it as a D-by-out_dim matrix (numpy array).
            The m-th column should correspond to the m-th largest eigenvalue of the covariance matrix.
            Each column of W must have a unit L2 norm.
    Todo:
        1. build mu
        2. build the covariance matrix Sigma: a D-by-D matrix (numpy array).
        3. We have provided code of how to compute W from Sigma
    Useful tool:
        1. np.mean: find the mean vector
        2. np.matmul: for matrix-matrix multiplication
        3. the builtin "reshape" and "transpose()" function of a numpy array
    """

    X = np.copy(X)
    D = X.shape[0] # feature dimension
    N = X.shape[1] # number of data instances

    ### Your job  starts here ###
    """
        use the following:
        np.linalg.eigh (or np.linalg.eig) for eigendecomposition. it returns
        V: eigenvalues, W: eigenvectors
        This function has already L2 normalized each eigenvector.
        NOYE: the output may be complex value: do .real to keep the real part of V and W
        sort the eigenvectors by sorting corresponding eigenvalues
        return mu and W
    
    """
    
    
    
    return mu, W

    ### Your job  ends here ###


### Your job  starts here ###   
"""
    load MNIST
    compute PCA
    produce figures and plots
"""





