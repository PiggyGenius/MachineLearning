#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import sys
sys.path.insert(0,"../../Tools")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mltools import plot_decision_regions
from sklearn.svm import SVC 

# Seed the random generator
np.random.seed(0)

# We create a random array [200,2] from standard distribution
data = np.random.randn(200, 2)

# If both features are greater than 0, class is one, else it is -1
classes = np.logical_xor(data[:, 0] > 0, data[:, 1] > 0)
classes = np.where(classes, 1, -1)

# We train a kernel SVM on the data
svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(data, classes)

# Plotting decison region
plot_decision_regions(data, classes, classifier=svm)
plt.legend(loc='upper left')
plt.show()
