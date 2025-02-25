import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from PIL import Image


def PCA(X, out_dim):
    np.set_printoptions(threshold=np.inf) 
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
    D = X.shape[0] # feature dimension (rows)
    N = X.shape[1] # number of data instances (columns)
    # print(D,N)

    ### Your job  starts here ###

    mu = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mu
    ## builds covariance matrix Sigma
    Sigma = np.cov(X_centered, bias=True, rowvar=True)
    # non_zero_rows = np.any(Sigma != 0, axis=1)

    # print(non_zero_rows)

    # Perform eigendecomposition
    V, W = np.linalg.eigh(Sigma)
    V, W = V.real, W.real
    # print(W[:,-(out_dim)].reshape(-1,1))
    # print(V[-(out_dim)])
    # sorted_indices = np.argsort(V)[::-1]
    # print(V[sorted_indices[0]])
    # print(W[:,sorted_indices[0]].reshape(-1,1))
    W = W[:,-out_dim:]


    
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


def reconstr(images,W_list,mu_list,dims):
    #center data again
    # mu = np.mean(images, axis=1, keepdims=True)
    # images = images - mu
    reconstructions = {}
    #reconstruct to each dimension
    for d in dims:
        W = W_list[d]  # Get PCA projection matrix for current dimension
        proj = W.T @ (images - mu_list[d]) # Project onto PCA subspace
        reconstructions[d] = (W @ proj) + mu_list[d]  # Reconstruct image

    return reconstructions

def img_plt(og_image, reconstr, dims):
    i = 1
    plt.figure(figsize=(10,6))
    plt.subplot(1,6,i)
    plt.imshow(og_image.reshape(28,28))
    plt.title("Original")
    plt.xticks([])
    plt.yticks([])
    for d in dims:
        i = i +1
        plt.subplot(1,6,i)
        plt.imshow(reconstr[d].reshape(28,28))
        plt.title(f"{d} Image")
        plt.xticks([])
        plt.yticks([])
    plt.show()

def multPCA(X,dims):
    W_list, mu_list = {},{}

    for d in dims:
        mu_list[d],W_list[d] = PCA(X, d)  # Compute PCA projection matrix W
    return W_list,mu_list

def mse_error(original, reconstr):
    return np.mean((original - reconstr)**2)

def dim_errors(test_data, reconstructions, dimesions):
    errors = {}
    for d in dimesions:
        recon = reconstructions[d]
        error = mse_error(test_data, recon)
        errors[d] = error
    return errors



### Your job  starts here ###   
"""
    load MNIST
    compute PCA
    produce figures and plots
"""

# Load MNIST data
mnist = pd.read_csv('PA2/mnist_test.csv').values
np.set_printoptions(threshold=np.inf) 
# Extract images of digit 3
#Each row is a picture (sample)
#each column is a pixel value (feature)
digit_3 = mnist[mnist[:, 0] == 3][:, 1:]  # Select rows where label == 3, remove labels
print("Digit 3 dataset shape:", digit_3.shape)

#Transpose so that we have shape (D=784, N) instead of (N, D)
#Each column is a picture (sample)
#each row is a pixel value (feature)
digit_3 = digit_3.T

# Print dataset shape
print("Digit 3 dataset shape:", digit_3.shape)  # (784, N)

dims = [2, 8, 64, 128, 784]

#Pick one image and apply PCA on all dimensions
str_indx = 9
end_indx = 10

image = digit_3[:,str_indx:end_indx].reshape(-1, end_indx - str_indx)
#print(image)

W_list, mu_list = multPCA(digit_3,dims)

# for d in dims:
#     mu_list[d],W_list[d] = PCA(digit_3, d)  # Compute PCA projection matrix W




# # test_img = digit_3[:, 0].reshape(-1, 1)  # Select first column and reshape
#plot original image

#gets the reconstructions
reconstructions = reconstr(image,W_list,mu_list,dims)
#plots them for a single image ast different dimensionalities
img_plt(image,reconstructions,dims)
#reconstruct to each dimension
# for d in dims:
#     i = i +1
#     W = W_list[d]  # Get PCA projection matrix for current dimension
#     proj = W.T @ images  # Project onto PCA subspace
#     reconstructions[d] = (W @ proj) + mu_list[d]  # Reconstruct image
#     #plot each dimension
#     plt.subplot(1,6,i)
#     plt.imshow(reconstructions[d].reshape(28,28))
#     plt.title(f"{d} Image")
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

# (mu, W) = PCA(images,end_indx - str_indx)
dimensions = [x for x in range(10, 785, 10)]
#initalize test data (100 3 images)
test_data = digit_3[:,350:450]
#initialize training data for 8 
digit_8 =  mnist[mnist[:, 0] == 8][:, 1:]
digit_8 = digit_8.T
#initialize training data for 9 
digit_9 =  mnist[mnist[:, 0] == 9][:, 1:]
digit_9 = digit_9.T

#training data for 3 and 8 
training_3_8= np.append(digit_3,digit_8,axis=1)
#training data for 3 and 8 and 9
training_3_8_9 = np.append(training_3_8,digit_9,axis=1)

#Apply PCA on all 3 training sets
#Already have training set 3s from previous calculation 
W3_list,mu3_list = multPCA(digit_3,dimensions)
#reconstruct test data using training data
#Project test data onto training set 3s PCA Space
recon_3 = reconstr(test_data,W3_list,mu3_list,dimensions)
#compute training 3 and 8
W3_8_list,mu3_8_list = multPCA(training_3_8,dimensions)
#reconstruct test data using training data
#Project test data onto training set 3_8s PCA Space
recon_3_8= reconstr(test_data,W3_8_list,mu3_8_list,dimensions)
#compute training 3 and 8 and 9
W3_8_9_list,mu3_8_9_list = multPCA(training_3_8_9,dimensions)
#reconstruct test data using training data
#Project test data onto training set 3_8_9s PCA Space
recon_3_8_9= reconstr(test_data,W3_8_9_list,mu3_8_9_list,dimensions)
#calculate 3 errors FOR EACH DIMENSION
errors_3 = dim_errors(test_data,recon_3,dimensions)
errors_3_8 = dim_errors(test_data,recon_3_8,dimensions)
errors_3_8_9 = dim_errors(test_data,recon_3_8_9,dimensions)

# Extract X (dimensions) and Y (error values) for each dataset
x_values = dimensions
y_errors_3 = [errors_3[d] for d in dimensions]
y_errors_3_8 = [errors_3_8[d] for d in dimensions]
y_errors_3_8_9 = [errors_3_8_9[d] for d in dimensions]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each error curve
plt.plot(x_values, y_errors_3, label="Training on Digit 3", marker='o')
plt.plot(x_values, y_errors_3_8, label="Training on Digit 3 & 8", marker='s')
plt.plot(x_values, y_errors_3_8_9, label="Training on Digit 3, 8 & 9", marker='^')

# Labels and title
plt.xlabel("PCA Dimensions")
plt.ylabel("Reconstruction Error (MSE)")
plt.title("Reconstruction Error vs PCA Dimensions")
plt.legend()
plt.grid(True)

# Show the plot
plt.show()