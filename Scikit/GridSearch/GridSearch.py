#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

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
# We set the param_grid parameter to a list of dictionaries to specify the parameters that we'd want to tune
# Here we tune both the C and gamma parameter
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
param_grid = [{'svc__C': param_range, 'svc__kernel': ['linear']},
              {'svc__C': param_range, 'svc__gamma': param_range, 'svc__kernel': ['rbf']}]
gs = GridSearchCV(pipeline, param_grid, scoring = 'accuracy', cv = 10, n_jobs = -1)
gs = gs.fit(train_values, train_classes)

# We print the accuracy best accuracy score
print('Best accuracy score: {}'.format(gs.best_score_))
print('Best accuracy parameters: {}'.format(gs.best_params_))

# Now we use the independent test dataset to estimate the performance of the best selected model
svc = gs.best_estimator_
svc.fit(train_values, train_classes)
print('\nTest accuracy: {:.3f}'.format(svc.score(test_values,test_classes)))
