#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import sys
sys.path.insert(0,"../../Tools/")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mltools import plot_decision_regions

# The name of the flowers is already loaded as integers for better performance
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

# We create a svm model and train it on the train data
svm = SVC(kernel='linear',C=1.0,random_state = 0)
svm.fit(X_train_std,y_train)

X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
