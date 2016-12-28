#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from matplotlib.ticker import FormatStrFormatter
from RBFKernelPCA import RBF_Kernel_PCA

# We create a dataset of two half shapes moons
values, classes = make_moons(n_samples = 100, random_state = 123)
plt.scatter(values[classes == 0, 0], values[classes == 0, 1], color = 'red', marker = '^', alpha = 0.5)
plt.scatter(values[classes == 1, 0], values[classes == 1, 1], color = 'blue', marker = 'o', alpha = 0.5)
plt.show()

# These two half-moon shapes are not linearly seperable and our goal is to unfold the half-moons via kernel PCA so that the datasets can be used with linear classifiers.
# First, we see what the datasets looks like if we project it onto the principal components via standard PCA.

# We use PCA on the dataset
pca = PCA(n_components = 2)
pca_values = pca.fit_transform(values)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7,3))

# We display the data with a shift, to better visualize the class overlap
ax[0].scatter(pca_values[classes == 0, 0], pca_values[classes == 0, 1], color = 'red', marker = '^', alpha = 0.5)
ax[0].scatter(pca_values[classes == 1, 0], pca_values[classes == 1, 1], color = 'blue', marker = 'o', alpha = 0.5)
ax[1].scatter(pca_values[classes == 0, 0], np.zeros((50,1)) + 0.02, color = 'red', marker = '^', alpha = 0.5)
ax[1].scatter(pca_values[classes == 1, 0], np.zeros((50,1)) - 0.02, color = 'blue', marker = 'o', alpha = 0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.show()

# Now we use kernel PCA
kernel_pca_values = RBF_Kernel_PCA(values, gamma = 15, n_components = 2)
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7,3))
ax[0].scatter(kernel_pca_values[classes == 0, 0], kernel_pca_values[classes == 0, 1], color = 'red', marker = '^', alpha = 0.5)
ax[0].scatter(kernel_pca_values[classes == 1, 0], kernel_pca_values[classes == 1, 1], color = 'blue', marker = 'o', alpha = 0.5)
ax[1].scatter(kernel_pca_values[classes == 0, 0], np.zeros((50,1)) + 0.02, color = 'red', marker = '^', alpha = 0.5)
ax[1].scatter(kernel_pca_values[classes == 1, 0], np.zeros((50,1)) - 0.02, color = 'blue', marker = 'o', alpha = 0.5)
ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1, 1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
ax[0].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
ax[1].xaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
plt.show()
