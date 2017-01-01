#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score

# We retrieve the Breast Cancer Wisconsin dataset
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

# We fit the model on the data and predict the classes of the test dataset
pipeline.fit(train_values, train_classes)
predicted_classes = pipeline.predict(test_values)

# We print the confusion_matrix of our model
conf_matrix = confusion_matrix(y_true = test_classes, y_pred = predicted_classes)
print(conf_matrix)

# Nicer display of confusion matrix
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.matshow(conf_matrix, cmap = plt.cm.Blues, alpha = 0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x = j, y = i, s = conf_matrix[i, j], va = 'center', ha = 'center')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

print('Precision: {:.3f}'.format(precision_score(y_true = test_classes, y_pred = predicted_classes)))
print('Recall: {:.3f}'.format(recall_score(y_true = test_classes, y_pred = predicted_classes)))
print('F1: {:.3f}'.format(f1_score(y_true = test_classes, y_pred = predicted_classes)))

###########################################################################################
#
# Remember that the positive class in scikit-learn is the class that is labeled as class 1.
# If we want to specify a different positive label, we can construct our own scorer via
# the make_scorer function, which we can then directly provide as an argument to the
# scoring parameter in GridSearchCV.
# In python: 
#   scorer = make_scorer(f1_score, pos_label=0)
#   gs = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scorer, cv=10)
#
###########################################################################################
