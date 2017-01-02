#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import sys
sys.path.insert(0, "../MajorityVoteClassifier")
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
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from MajorityVoteClassifier import MajorityVoteClassifier
from itertools import product

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

# Now we plot the ROC curves of each classifier with their auc score on the test set now
colors = ['red', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    predicted_classes = clf.fit(train_values, train_classes).predict_proba(test_values)[:, 1]
    FPR, TPR, thresholds = roc_curve(y_true = test_classes, y_score = predicted_classes)
    roc_auc = auc(x = FPR, y = TPR)
    plt.plot(FPR, TPR, color = clr, linestyle = ls, label = '{} (auc = {:.2f}'.format(label, roc_auc))

# We also plot the curve corresponding to random predictions
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle = '--', color = 'gray', linewidth = 2)
plt.xlim([-0.1, 1.1])
plt.ylim([-0.1, 1.1])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print('We see a lower AUC value on the test set with KNN which is a tell of overfitting.\n')


###########################################################################################
#
# Since we only selected two features for the classification examples, it would be
# interesting to see what the decision region of the ensemble classifier actually
# looks like. Although it is not necessary to standardize the training features prior
# to model fitting because our logistic regression and k-nearest neighbors pipelines
# will automatically take care of this, we will standardize the training set so that the
# decision regions of the decision tree will be on the same scale for visual purposes.
#
###########################################################################################

# We standardize the data
ss = StandardScaler()
std_train_values = ss.fit_transform(train_values)

# We compute the range of the values
x_min = std_train_values[:, 0].min() - 1
x_max = std_train_values[:, 0].max() + 1
y_min = std_train_values[:, 1].min() - 1
y_max = std_train_values[:, 1].max() + 1

# We compute a matrix of values to get the decision region
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
f, axarr = plt.subplots(nrows = 2, ncols = 2, sharex = 'col', sharey = 'row', figsize = (12, 10))

# Now we compute and plot the decision region of each classifier
# product creates an iterable object consisting of cartesian product tuple.
# product([0, 1], [0, 1]) = [(0, 0), (0, 1), (1, 0), (1, 1)]
for idx, clf, la in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    # We fit the classifiers on the training data
    clf.fit(std_train_values, train_classes)
    # Now we predict on our matrix to get the decision region
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # We plot class 0 in blue and class 1 in red
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha = 0.3)
    axarr[idx[0], idx[1]].scatter(std_train_values[train_classes == 0, 0], std_train_values[train_classes == 0, 1], c = 'blue', marker = '^', s = 50)
    axarr[idx[0], idx[1]].scatter(std_train_values[train_classes == 1, 0], std_train_values[train_classes == 1, 1], c = 'red', marker = 'o', s = 50)
    axarr[idx[0], idx[1]].set_title(la)
plt.text(-3.5, -4.5, s = 'Sepal width [standardized]', ha = 'center', va = 'center', fontsize = 12)
plt.text(-10.5, 4.5, s = 'Petal length [standardized]', ha = 'center', va = 'center', fontsize = 12, rotation = 90)
plt.show()

# Now we can tune the inverse regularization paramter c of the logic regression classifier as well as the decision tree depth via grid search
# To know how to access the parameters: print(mv_clf.get_params())
params = {'decisiontreeclassifier__max_depth': [1, 2], 'pipeline-1__clf__C': [0.001, 0.1, 100.0]}
grid = GridSearchCV(estimator = mv_clf, param_grid = params, cv = 10, scoring = 'roc_auc')
grid.fit(train_values, train_classes)

# We print the different hyperparameter value combinations and the average ROC AUC scores computed via 10-fold cross-validation
print('Grid search on logistic regression and decision tree classifier:')
cv_keys = ('mean_test_score', 'std_test_score', 'params')
for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print('{:.3f} +/- {:.2f} {}'.format(grid.cv_results_[cv_keys[0]][r], grid.cv_results_[cv_keys[1]][r]/ 2.0, grid.cv_results_[cv_keys[2]][r]))
print('\nBest parameters: {}'.format(grid.best_params_))
print('Accuracy: {:.2f}'.format(grid.best_score_))
