#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.decomposition import KernelPCA

# We create the dataset and project our values with kernel PCA
values, classes = make_moons(n_samples = 100, random_state = 123)
kernel_pca = KernelPCA(n_components = 2, kernel = 'rbf', gamma = 15)
kernel_pca_values = kernel_pca.fit_transform(values)

# We plot the projected dataset
plt.scatter(kernel_pca_values[classes == 0, 0], kernel_pca_values[classes == 0, 1], color = 'red', marker = '^', alpha = 0.5)
plt.scatter(kernel_pca_values[classes == 1, 0], kernel_pca_values[classes == 1, 1],color = 'blue', marker = 'o', alpha = 0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# More on nonlinear dimensionality reduction: http://scikit-learn.org/stable/modules/manifold.html
