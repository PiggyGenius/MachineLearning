#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from LogisticRegression import LogisticRegression
from plot import plot_decision_regions

# The name of the flowers is already loaded as integers for better performance
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

# We create a sigmoid model and train it on the train data
lr = LogisticRegression(0.01,50)
lr.fit(train_values_std,train_classes)
print(lr.predict_proba(test_values_std[0,:]))


# Predictions of the perceptron and mistakes
predicted_classes = lr.predict(test_values_std)
print("Misclassifed samples: %d" % (test_classes != predicted_classes).sum())
print("Accuracy: %.2f" % accuracy_score(test_classes,predicted_classes))

# We combine the test and train sets then we will specify test range in test_idx
combined_values_std = np.vstack((train_values_std,test_values_std))
combined_classes = np.hstack((train_classes,test_classes))
plot_decision_regions(combined_values_std,combined_classes,classifier=lr,test_start=105)
plt.xlabel("petal length [standardized]")
plt.ylabel("petal width [standardized]")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()
