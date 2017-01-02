#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from MajorityVoteClassifier import MajorityVoteClassifier

# We load the iris dataset and constrain ourselves to two features and two class
iris = datasets.load_iris()
values, classes = iris.data[50:, [1, 2]], iris.target[50:]
le = LabelEncoder()
classes = le.fit_transform(classes)

# We split the dataset
train_values, test_values, train_classes, test_classes = train_test_split(values, classes, test_size = 0.5, random_state = 1)

# We train three classifiers on data and look at individual performance with K-folds
clf_1 = LogisticRegression(penalty = 'l2', C = 0.001, random_state = 0)
clf_2 = DecisionTreeClassifier(max_depth = 1, criterion = 'entropy', random_state = 0)
clf_3 = KNeighborsClassifier(n_neighbors = 1, p = 2, metric = 'minkowski')

# We construct two pipelines, no need to standardize data with decision tree
pipeline_1 = Pipeline([['sc', StandardScaler()], ['clf', clf_1]])
pipeline_3 = Pipeline([['sc', StandardScaler()], ['clf', clf_3]])

# We fit our classifiers and compute the individual accuracy scores
clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']
print('10-fold cross-validation on individual classifiers:')
for clf, label in zip([pipeline_1, clf_2, pipeline_3], clf_labels):
    scores = cross_val_score(estimator = clf, X = train_values, y = train_classes, cv = 10, scoring = 'roc_auc')
    print('ROC AUC: {:.2f} +/- {:.2f} [{}]'.format(scores.mean(), scores.std(), label))

# Now we combine the individual classifiers for majority rule voting
mv_clf = MajorityVoteClassifier(classifiers = [pipeline_1, clf_2, pipeline_3])
clf_labels += ['Majority voting']
all_clf = [pipeline_1, clf_2, pipeline_3, mv_clf]
print('\n10-fold cross-validation on individual classifiers and majority voting:')
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator = clf, X = train_values, y = train_classes, cv = 10, scoring = 'roc_auc')
    print('ROC AUC: {:.2f} +/- {:.2f} [{}]'.format(scores.mean(), scores.std(), label))
