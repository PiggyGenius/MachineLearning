#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
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

# We get the covariance matrix of the standardized train set
covariance_matrix = np.cov(std_train_values.T)
# We need the eigenpairs of the covariance matrix to get the principal components
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
print('Eigenvalues \n%s'%eigenvalues,'\n')

# We calculate the explained variance to know how import are the n first components
eigen_sum = sum(eigenvalues)
explained_variance = [(i/eigen_sum) for i in sorted(eigenvalues,reverse = True)]
cumulative_explained_variance = np.cumsum(explained_variance)

plt.bar(range(1,14),explained_variance,alpha=0.5,align='center',label='explained variance')
plt.step(range(1,14),cumulative_explained_variance,where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc = 'best')
plt.show()

# Now we are going to select the first and second most important components, 60% of the variance in the data
eigenpairs = [(np.abs(eigenvalues[i]),eigenvectors[:,i]) for i in range(len(eigenvalues))]
eigenpairs.sort(reverse = True)
projection_matrix = np.hstack((eigenpairs[0][1][:,np.newaxis],eigenpairs[1][1][:,np.newaxis]))
print('Projection matrix:\n',projection_matrix,'\n')

# We can now convert the data
print(std_train_values[0].dot(projection_matrix))
pca_train_values = std_train_values.dot(projection_matrix)

# We now visualize the wine data stored as a 124*2 dimensional matrix

colors = ['r', 'b', 'g']
markers = ['s', 'x', 'o']
for l, c, m in zip(np.unique(train_classes),colors,markers):
    plt.scatter(pca_train_values[train_classes == l,0], pca_train_values[train_classes == l,1],c = c,label = l,marker = m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.show()
