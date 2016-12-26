#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class SBS():

    def __init__(self, estimator, k_features, scoring = accuracy_score, test_size = 0.25, random_state=1):
        """Sequential feature selection
        k_features is the desired number of features
        """
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, values, classes):
        """Reduce feature dimensionality
        We are fed train_set and still split it again, valdiation set
        """
        train_values, test_values, train_classes, test_classes = train_test_split(values, classes, test_size = self.test_size,random_state = self.random_state)
        dim = train_values.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(train_values, train_classes, test_values, test_classes, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r = dim-1):
                score = self._calc_score(train_values, train_classes,test_values, test_classes, p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, values):
        return values[:, self.indices_]

    def _calc_score(self, train_values, train_classes,test_values, test_classes, indices):
        self.estimator.fit(train_values[:, indices], train_classes)
        classes_prediction = self.estimator.predict(test_values[:, indices])
        score = self.scoring(test_classes, classes_prediction)
        return score
