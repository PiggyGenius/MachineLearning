#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import sys
sys.path.insert(0,"../../Tools")
import numpy as np
import matplotlib.pyplot as plt
from mltools import plot_decision_regions
from sklearn import datasets
from sklearn.svm import SVC 
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

# We load the iris data
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

# We split the X and y arrays into 30% test data and 70% training data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

# For better performance we standardize the data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# fitting on iris data with small gamma value, small cut-off value
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)

# We combine the test and train sets then we will specify test range in test_idx
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

# We plot the decision regions
plt.figure(1)
plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')

# New we fit the iris data with large gamma value and have overfitting
svm = SVC(kernel='rbf', random_state=0, gamma=100.0, C=1.0)
svm.fit(X_train_std, y_train)

# We plot the decision region
plt.figure(2)
plot_decision_regions(X_combined_std,y_combined, classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
