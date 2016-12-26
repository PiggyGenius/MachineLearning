#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Getting data
data_file = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header = None)

# We split into train and test sets
values = data_file.iloc[:,1:].values
classes = data_file.iloc[:,0].values
train_values,test_values,train_classes,test_classes = train_test_split(values,classes,test_size = 0.3,random_state = 0)

# We standardize the data
std_scaler = StandardScaler()
std_train_values = std_scaler.fit_transform(train_values)
std_test_values = std_scaler.fit_transform(test_values)

# We calculate the mean vectors
np.set_printoptions(precision = 4)
mean_vectors = []
min_class = min(classes)
max_class = max(classes) + 1
for label in range(min_class,max_class):
    mean_vectors.append(np.mean(std_train_values[train_classes == label],axis = 0))
    print('Mean vector {}: {}\n'.format(label,mean_vectors[label-1]))

# We compute the within scatter matrix 
dimension = data_file.count().size - 1
within_matrix = np.zeros((dimension,dimension))
for label, mv in zip(range(min_class,max_class),mean_vectors):
    class_scatter = np.cov(std_train_values[train_classes == label].T)
    within_matrix += class_scatter
print('Within-class scatter matrix: {}x{}'.format(within_matrix.shape[0],within_matrix.shape[1]),'\n')

# Our training dataset is not uniformly distributed (class-wise)
print('Class distribution: {}'.format(np.bincount(train_classes)[1:]),'\n')

# We compute the bewteen scatter matrix
mean_overall = np.mean(std_train_values,axis = 0)
mean_overall = mean_overall.reshape(dimension,1)
between_matrix = np.zeros((dimension,dimension))
for i, mean_vector in enumerate(mean_vectors):
    n = train_values[train_classes == i+1,:].shape[0]
    mean_vector = mean_vector.reshape(dimension,1)
    between_matrix += n * (mean_vector - mean_overall).dot((mean_vector - mean_overall).T)
print('Between-class scatter matrix: {}x{}'.format(between_matrix.shape[0],between_matrix.shape[1]),'\n')

# We compute the eigenpairs of the inverse of the within matrix times the between matrix
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(within_matrix).dot(between_matrix))
eigenpairs = [(np.abs(eigenvalues[i]),eigenvectors[:,i]) for i in range(len(eigenvalues))]
eigenpairs = sorted(eigenpairs,key = lambda k: k[0],reverse = True)
print('Eigenvalues in decreasing order:')
for eigenvalue in eigenpairs:
    print(eigenvalue[0])
print('\n')

# We plot the dicriminability captured by the linear discriminants (eigenvectors)
total_sum = sum(eigenvalues.real)
discriminants = [(i / total_sum) for i in sorted(eigenvalues.real, reverse = True)]
cumulative_discriminants = np.cumsum(discriminants)
plt.bar(range(1, 14), discriminants, alpha = 0.5, align = 'center', label = 'individual discriminability')
plt.step(range(1, 14), cumulative_discriminants, where = 'mid', label = 'cumulative "discriminability"')
plt.ylabel('discriminability ratio')
plt.xlabel('Linear Discriminants')
plt.ylim([-0.1, 1.1])
plt.legend(loc = 'best')
plt.show()

# We create the transformation matrix using the two most discriminative eigenvectors
transformation_matrix = np.hstack((eigenpairs[0][1][:,np.newaxis].real,eigenpairs[1][1][:,np.newaxis].real))
print('Transformation matrix: \n',transformation_matrix)

# Finally we transform the training data and plot the result
lda_train_values = std_train_values.dot(transformation_matrix)
colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(train_classes), colors, markers):
    plt.scatter(lda_train_values[train_classes == l, 0], lda_train_values[train_classes == l, 1], c = c, label = l, marker = m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='upper right')
plt.show()
