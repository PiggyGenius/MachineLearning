#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0,"../../Tools/")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from plot import plot_decision_regions

# Getting data
data_file = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header = None)

# We split into train and test sets
values = data_file.iloc[:,1:].values
classes = data_file.iloc[:,0].values
train_values,test_values,train_classes,test_classes = train_test_split(values,classes,test_size = 0.3,random_state = 0)

# We standardize the data
std_scaler = StandardScaler()
std_train_values = std_scaler.fit_transform(train_values)
std_test_values = std_scaler.fit_transform(test_values)

# We transform the data with LDA
lda = LinearDiscriminantAnalysis(n_components = 2)
lda_train_values = lda.fit_transform(std_train_values,train_classes)

# We use logistic regression
lr = LogisticRegression()
lr = lr.fit(lda_train_values, train_classes)

# We plot the train dataset
plot_decision_regions(lda_train_values, train_classes, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

# We plot the test dataset
lda_test_values = lda.transform(std_test_values)
plot_decision_regions(lda_test_values, test_classes, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.show()

