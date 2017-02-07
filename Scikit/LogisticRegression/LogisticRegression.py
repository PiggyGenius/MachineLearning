#!/usr/bin/python2.7
# -*-coding:Utf-8 -*
import sys
sys.path.insert(0,"../../Tools/")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

# We create a sigmoid model and train it on the train data
lr = LogisticRegression(C = 1000.0,random_state = 0)
lr.fit(X_train_std,y_train)

# Predictions of the perceptron and mistakes
y_pred = lr.predict(X_test_std)
print("Misclassifed samples: %d" % (y_test != y_pred).sum())
print("Accuracy: %.2f" % accuracy_score(y_test,y_pred))

# We combine the test and train sets then we will specify test range in test_idx
X_combined_std = np.vstack((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))

plot_decision_regions(X_combined_std,y_combined,classifier=lr,test_idx=range(105,150))
plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
