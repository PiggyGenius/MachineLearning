#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import sys
sys.path.insert(0,"../../Tools/")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from plot import plot_decision_regions
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

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

# We use PCA to transform the data and then use logistic regression
pca = PCA(n_components=2)
lr = LogisticRegression()
pca_train_values = pca.fit_transform(std_train_values)
pca_test_values = pca.transform(std_test_values)
lr.fit(pca_train_values, train_classes)

# We plot the result on our train set and test set
plot_decision_regions(pca_train_values, train_classes, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

plot_decision_regions(pca_test_values, test_classes, classifier=lr)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.show()

# If we want to print explained variance ratio
pca = PCA(n_components = None) # No dimenstionality reduction when set to None
pca_train_values = pca.fit_transform(std_train_values)
print('Explained variance ratio:\n',pca.explained_variance_ratio_)
