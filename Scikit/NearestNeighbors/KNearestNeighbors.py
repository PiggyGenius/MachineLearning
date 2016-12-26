#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import sys
sys.path.insert(0,"../../Tools")
import numpy as np
import matplotlib.pyplot as plt
from mltools import plot_decision_regions
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# We load the iris data
iris = datasets.load_iris()
values = iris.data[:,[2,3]]
classes = iris.target

# We split the X and y arrays into 30% test data and 70% training data
train_values,test_values,train_classes,test_classes = train_test_split(values,classes,test_size = 0.3,random_state = 0)

# For better performance we standardize the data
sc = StandardScaler()
sc.fit(train_values)
train_values_std = sc.transform(train_values)
test_values_std = sc.transform(test_values)

# fitting on iris data with minkowski norm which is p norm with p = 2, euclidian distance, 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
knn.fit(train_values_std, train_classes)

# We combine the test and train sets then we will specify test range in test_idx
X_combined_std = np.vstack((train_values_std,test_values_std))
y_combined = np.hstack((train_classes,test_classes))

# We plot the decision regions
plot_decision_regions(X_combined_std,y_combined,classifier=knn,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
