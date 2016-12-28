#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from scipy import exp

def RBF_Kernel_PCA(values, gamma, n_components):
    """
    RBF Kernel PCA implementation.

    Parameters
    ----------------------------------------
    values: {NumPy ndarray}, shape = [n_samples, n_features]
    gamma: float
        Tuning paramater of the RBF kernel
    n_components: int
        Number of principal comonents to return

    Returns
    ----------------------------------------
    pca_values: {NumPy ndarray}, shape = [n_samples, k_features]
        Projected dataset
    lambdas: list
        Eigenvalues
    """

    # Calculate pairwise squared Euclidian distances in the MxN dimensional dataset
    square_distance = pdist(values, 'sqeuclidean')

    # Convert pairwise distances into a square matrix
    square_distance_matrix = squareform(square_distance)

    # Compute the symmetric kernel matrix
    kernel = exp(-gamma * square_distance_matrix)

    # Center the kernel matrix
    N = kernel.shape[0]
    one_n = np.ones((N,N)) / N
    kernel = kernel - one_n.dot(kernel) - kernel.dot(one_n) + one_n.dot(kernel).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix, numpy.eigh returns them sorted
    eigenvalues, eigenvectors = eigh(kernel)

    # Collect the top K eigenvectors, projected samples
    pca_values = np.column_stack((eigenvectors[:,-i] for i in range(1,n_components + 1)))

    # Collect the corresponding eigenvalues
    lambdas = [eigenvalues[-i] for i in range(1,n_components + 1)]

    return pca_values, lambdas
