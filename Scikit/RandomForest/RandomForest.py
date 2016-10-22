#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import sys
sys.path.insert(0,"../../Tools")
import numpy as np
import matplotlib.pyplot as plt
from mltools import plot_decision_regions
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

# We load the iris data
iris = datasets.load_iris()
values = iris.data[:,[2,3]]
classes = iris.target

# We split the X and y arrays into 30% test data and 70% training data
train_values,test_values,train_classes,test_classes = train_test_split(values,classes,test_size = 0.3,random_state = 0)

# fitting on iris data with 10 decision trees, we parallelize the model on two cores
forest = RandomForestClassifier(criterion='entropy',n_estimators=10,random_state=1,n_jobs=2)
forest.fit(train_values, train_classes)

# We plot the decision regions
plot_decision_regions(values,classes,classifier=forest,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
