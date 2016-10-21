#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import sys
sys.path.insert(0,"../../Tools/")
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from mltools import plot_decision_regions

# The name of the flowers is already loaded as integers for better performance
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

# We split the X and y arrays into 30% test data and 70% training data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

# We create a decision tree model
tree = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
# We don't need feature scailing for decision tree
tree.fit(X_train,y_train)

# We combine the test and train sets then we will specify test range in test_idx
X_combined = np.vstack((X_train,X_test))
y_combined = np.hstack((y_train,y_test))

# Awesome feature using graphviz, we can export the decision tree
# Create the png with: dot -Tpng tree.dot -o tree.png
export_graphviz(tree,out_file='tree.dot',feature_names=['petal length','petal width'])

plot_decision_regions(X_combined, y_combined,classifier=tree,test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
