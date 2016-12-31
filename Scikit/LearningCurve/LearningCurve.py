#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve

# We retrieve the dataset
data_file = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

# We get the values, 30 features, and the two classes. We use LabelEncoder to encode B = 0 and M = 1.
values = data_file.loc[:, 2:].values
classes = data_file.loc[:, 1].values
le = LabelEncoder()
classes = le.fit_transform(classes)

# We devide the dataset into a train and test set
train_values, test_values, train_classes, test_classes = train_test_split(values, classes, test_size = 0.20, random_state = 1)

# We construct the pipeline
pipeline = Pipeline([('ss', StandardScaler()), ('lr', LogisticRegression(penalty = 'l2', random_state = 0))])

# We compute the learning curve on the pipeline
# We set train_sizes = linespace(0.1,1.0,10) tp use 10 evenly spaced relative intervals for the training set sizes
# By default the learning curve uses stratified k-fold cross-validation and we set k = 10 with cv = 10
train_sizes, train_scores, test_scores = learning_curve(pipeline, train_values, train_classes, train_sizes = np.linspace(0.1,1.0,10), cv = 10, n_jobs = 1)

# We compute the mean score and standard deviation
train_mean = np.mean(train_scores, axis = 1)
train_std = np.std(train_scores, axis = 1)
test_mean = np.mean(test_scores, axis = 1)
test_std = np.std(test_scores, axis = 1)

# We plot the training and validation accuracy
# Fill_between describes the variance of the estimate by adding and substracting the standard deviation to the mean
plt.plot(train_sizes, train_mean, color = 'blue', marker = 'o', markersize = 5, label = 'Training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')
plt.plot(train_sizes, test_mean, color = 'green', linestyle = '--', marker = 's', markersize = 5, label = 'Validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = 0.15, color = 'green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')
plt.ylim([0.8, 1.0])
plt.show()
