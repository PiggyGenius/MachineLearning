#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

###########################################################################################
#
# We will work with the Breast Cancer Wisconsin Dataset which contains 569 samples of
# malignant and benign tumor cells. The first two columns in the dataset store the unique
# IDs numbers of the samples and the corresponding diagnosis, M = malignant and B = benign.
# The columns 3 to 32 contain 30 real-value features that have computed from digitized
# images of the cell nuclei, which can be used to build a model to predict whether a tumor
# is benign or malignant.
#
###########################################################################################

# We retrieve the dataset
data_file = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header = None)

# We get the values, 30 features, and the two classes. We use LabelEncoder to encode B = 0 and M = 1.
values = data_file.loc[:, 2:].values
classes = data_file.loc[:, 1].values
le = LabelEncoder()
classes = le.fit_transform(classes)

# We devide the dataset into a train and test set
train_values, test_values, train_classes, test_classes = train_test_split(values, classes, test_size = 0.20, random_state = 1)

# Pipeline takes a list of tuples where the first value in each tuple is the identifier used to access individual objects in the pipeline
pipeline = Pipeline([('ss',StandardScaler()), ('pca',PCA(n_components = 2)), ('lr', LogisticRegression(random_state = 1))])
pipeline.fit(train_values,train_classes)
print('Test Accuracy: %.3f'%pipeline.score(test_values,test_classes))

# We create the k-fold cross-validation module for 10 folds on the train dataset
k_fold = StratifiedKFold(n_splits = 10, random_state = 1).split(train_values,train_classes)
scores = []
for k, (train,test) in enumerate(k_fold):
    pipeline.fit(train_values[train], train_classes[train])
    score = pipeline.score(train_values[test], train_classes[test])
    scores.append(score)
    print('Fold: {}, Class distribution: {}, Accuracy: {:.3f}'.format(k+1,np.bincount(train_classes[train]),score))
print('Cross-validation accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))

# We can use the k-fold cross-validation module of scikit-learn
scores = cross_val_score(pipeline, train_values, train_classes, cv = 10, n_jobs = 1)
print('\n','Sklearn Cross-validation scores: {}'.format(scores))
print('Sklearn cross-validation accuracy: {:.3f} +/- {:.3f}'.format(np.mean(scores), np.std(scores)))
