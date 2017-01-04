#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier

# We load the wine datafile and set the feature labels, we only consider two features: alcohol and hue
wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header = None)
wine_data.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
wine_data = wine_data[wine_data['Class label'] != 1]
classes = wine_data['Class label'].values
values = wine_data[['Alcohol', 'Hue']].values

# We encode the labels and split the data
le = LabelEncoder()
classes = le.fit_transform(classes)
train_values, test_values, train_classes, test_classes = train_test_split(values, classes, test_size = 0.40, random_state = 1)

# We create an ensemble of 500 decision tree stumps
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 1)
abc = AdaBoostClassifier(base_estimator = tree, n_estimators = 500, learning_rate = 0.1, random_state = 0)

# We compute the accuracy score of the decision tree stump
tree = tree.fit(train_values, train_classes)
predicted_train_classes = tree.predict(train_values)
predicted_test_classes = tree.predict(test_values)
train_accuracy = accuracy_score(train_classes, predicted_train_classes)
test_accuracy = accuracy_score(test_classes, predicted_test_classes)
print('Decision tree stump:\nTrain accuracy: {:.3f}\nTest accuracy: {:.3f}'.format(train_accuracy, test_accuracy))
print('As expected the decision tree stump has bad accuracy.\n')

# We compute the accuracy score of the AdaBoost model
abc = abc.fit(train_values, train_classes)
predicted_train_classes = abc.predict(train_values)
predicted_test_classes = abc.predict(test_values)
train_accuracy = accuracy_score(train_classes, predicted_train_classes)
test_accuracy = accuracy_score(test_classes, predicted_test_classes)
print('Decision tree:\nTrain accuracy: {:.3f}\nTest accuracy: {:.3f}'.format(train_accuracy, test_accuracy))
print('We introduced variance to reduce the model bias which translates to overfitting here.')

# Now we compare the decision regions between the two decision tree and the AdaBoost model
x_min = train_values[:, 0].min() - 1
x_max = train_values[:, 0].max() + 1
y_min = train_values[:, 1].min() - 1
y_max = train_values[:, 1].max() + 1
# np.arange(a,b,c) returns a vector containing every value from a to b with a step of c
# np.meshgrid(a,b) create two matrix M1:NxN and M2:MxM with N the dimension of vector a and M the one of vector b. M1 consists of N identical lines corresponding to a whereas M2 consists of M identical columns corresponding to b.
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows = 1, ncols = 2, sharex = 'col', sharey = 'row', figsize = (15,10))
for idx, clf, tt in zip([0, 1], [tree, abc], ['Decision tree stump', 'AdaBoost']):
    clf.fit(train_values, train_classes)
    # Ravel transform NxM matrix into a 1x(N*M) vector
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z takes the shape of the xx matrix
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx, yy, Z, alpha = 0.3)
    axarr[idx].scatter(train_values[train_classes == 0, 0], train_values[train_classes == 0, 1], c = 'blue', marker = '^')
    axarr[idx].scatter(train_values[train_classes == 1, 0], train_values[train_classes == 1, 1], c = 'red', marker = 'o')
    axarr[idx].set_title(tt)
axarr[0].set_ylabel('Alcohol', fontsize = 12)
axarr[0].set_xlabel('Hue', fontsize = 12)
axarr[1].set_ylabel('Alcohol', fontsize = 12)
axarr[1].set_xlabel('Hue', fontsize = 12)
plt.show()
