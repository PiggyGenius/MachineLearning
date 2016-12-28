#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from matplotlib.ticker import FormatStrFormatter
from RBFKernelPCA import RBF_Kernel_PCA

# We create a dataset of two half moons and project them on 1 dimensional space
values, classes = make_moons(n_samples = 100, random_state = 123)
kernel_pca_values, lambdas = RBF_Kernel_PCA(values, gamma = 15, n_components = 1)

# We consider that the 26th point is a new point to project
new_value = values[25]
print('New value: {}'.format(new_value))
original_projected_value = kernel_pca_values[25]
print('Original projection: {}'.format(original_projected_value))

# We define a projection function for new values
def project_value(new_value, values, gamma, kernel_pca_values, lambdas):
    pairwise_distances = np.array([np.sum((new_value - row)**2) for row in values])
    kernel = np.exp(-gamma * pairwise_distances)
    return kernel.dot(kernel_pca_values / lambdas)

# We use the projection to recalculate the projection of the 26th point
new_projected_value = project_value(new_value, values, 15, kernel_pca_values, lambdas)
print('New projection: {}'.format(new_projected_value))

# Now we visualize the projection on the first principal components
plt.scatter(kernel_pca_values[classes == 0, 0], np.zeros((50)), color = 'red', marker = '^',alpha = 0.5)
plt.scatter(kernel_pca_values[classes == 1, 0], np.zeros((50)), color = 'blue', marker = 'o', alpha = 0.5)
plt.scatter(original_projected_value, 0, color = 'black', label = 'original projection of point X[25]', marker = '^', s = 100)
plt.scatter(new_projected_value, 0, color = 'green', label = 'remapped point X[25]', marker = 'x', s = 500)
plt.legend(scatterpoints = 1)
plt.show()
