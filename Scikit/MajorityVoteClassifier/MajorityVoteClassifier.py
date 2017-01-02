#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import six
from sklearn.base import clone
from sklearn.pipeline import _name_estimators

# We use BaseEstimator and ClassifierMixin to get some base functionality for free
# Such as the get_params and set_params and the score method for the prediction accuracy
# We import six to make the MajorityVoteClassifier compatible with python 2.7
class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """ Majority vote ensemble classifier
    Parameters
    ----------
    classifiers: array-like, shape = [n_classifiers]
        Different classifiers for the ensemble
    vote: str, {'classlabel', 'probability'}
        Default: 'classlabel' (recommended for calibrated classifiers)
        If vote = 'classlabel', the prediction is based on argmax of classlabels.
        Else if vote = 'probability', the argmax of the sum of probabilities is used.
    weights: array-like, shape = [n_classifiers]
        Optional, default: None
        If a list of values are provided, the classifers are weighted by importance.
    """

    def __init__(self, classifiers, vote = 'classlabel', weights = None):
        # _name_estimators generates names for a given array of estimators
        # for key, value in ... gives us the generated name as key and classifier as value
        self.named_classifiers = {key:value for key,value in _name_estimators(classifiers)}
        self.classifiers = classifiers
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        """ Fit classifiers
        Parameters
        ----------
        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.
        y: array-like, shape = [samples]
            Vector of target class labels.
        Returns
        -------
        self: object
        """
        # We use LabelEncoder to ensure class labels start with 0
        # It is important for np.argmax call in self.predict
        self.labelencoder_ = LabelEncoder()
        self.labelencoder_.fit(y)
        self.classes_ = self.labelencoder_.classes_
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labelencoder_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self

    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Matrix of training samples.
        Returns
        -------
        maj_vote : array-like, shape = [n_samples]
            Predicted class labels.
        """
        # We use the self.predict_proba method in case of probability vote
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            # We compute the prediction of each classifier for the X matrix
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            # lambda declares an argmax function that will be applied to the set with np
            # The np function applies the majority voting function on the predictions
            maj_vote = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights = self.weights)), axis = 1, arr = predictions)

        # We restore the original class labels and return the results of majority voting
        maj_vote = self.labelencoder_.inverse_transform(maj_vote)
        return maj_vote

    def predict_proba(self, X):
        """ Predict class probabilities for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors. n_samples: number of samples, n_features: number of features.
        Returns
        -------
        avg_proba : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis = 0, weights = self.weights)
        return avg_proba

    def get_params(self, deep = True):
        """ Get classifier parameter names for GridSearch"""
        if not deep:
            return super(MajorityVoteClassifier, self).get_params(deep = False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep = True)):
                    out['%s__%s' % (name, key)] = value
            return out
