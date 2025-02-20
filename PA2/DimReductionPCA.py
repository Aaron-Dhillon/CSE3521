import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd


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

    mu = np.mean(X, axis=1, keepdims=True)
    ## builds covariance matrix Sigma
    Sigma = np.cov(X, bias=True)

    # Perform eigendecomposition
    V, W = np.linalg.eigh(Sigma)
    V, W = V.real, W.real
    sorted_indices = np.argsort(V)[::-1]
    W = W[:, :out_dim]


    
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

# Load MNIST data
mnist = pd.read_csv('PA2/mnist_test.csv').values
# Extract images of digit 3
digit_3 = mnist[mnist[:, 0] == 3][:, 1:]  # Select rows where label == 3, remove labels

# Transpose so that we have shape (D=784, N) instead of (N, D)
digit_3 = digit_3.T

# Print dataset shape
print("Digit 3 dataset shape:", digit_3.shape)  # (784, N)

dims = [2, 8, 64, 128, 784]
W_list, mu_list = {},{}

for d in dims:
    mu_list[d],W_list[d] = PCA(digit_3, d)  # Compute PCA projection matrix W

test_img = digit_3[:, 0].reshape(-1, 1)  # Select first column and reshape

reconstructions = {}

for d in dims:
    W = W_list[d]  # Get PCA projection matrix for current dimension
    proj = W.T @ test_img  # Project onto PCA subspace
    reconstructions[d] = W @ proj  # Reconstruct image
