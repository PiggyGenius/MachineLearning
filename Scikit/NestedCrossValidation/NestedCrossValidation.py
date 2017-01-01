#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
pipeline = Pipeline([('ss', StandardScaler()), ('svc', SVC(random_state = 1))])

# We execute the grid search on the parameters of the SVC model
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
              {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}]
gs = GridSearchCV(pipeline, param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)

scores = cross_val_score(gs, values, classes, scoring = 'accuracy', cv = 5)
print('(SVM) Cross validation accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))

###########################################################################################
#
# The returned average cross-validation accuracy gives us a good estimate of what
# to expect if we tune the hyperparameters of a model and then use it on unseen data.
# For example, we can use the nested cross-validation approach to compare an
# SVM model to a simple decision tree classifier; for simplicity, we will only tune
# its depth parameter
#
###########################################################################################

gs = GridSearchCV(DecisionTreeClassifier(random_state = 0), param_grid = [{'max_depth': [1, 2, 3, 4, 5, 6, 7, None]}], scoring = 'accuracy', cv = 5)
scores = cross_val_score(gs, train_values, train_classes, scoring = 'accuracy', cv = 5)
print('(Decision tree) Cross validation accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))
