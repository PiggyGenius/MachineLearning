#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Retreiving data
wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
df_wine = pd.read_csv(wine_url,header=None)
columns = ["Class label","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
df_wine.columns = columns 

# Splitting data
classes =  df_wine.iloc[:, 0].values
values =  df_wine.iloc[:, 1:].values
train_values, test_values, train_classes, test_classes = train_test_split(values, classes, test_size = 0.3, random_state = 0)

feature_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators = 10000,random_state = 0,n_jobs = -1)
forest.fit(train_values, train_classes)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(train_values.shape[1]):
     print("%2d) %-*s %f" % (f + 1, 30,feature_labels[f],importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(train_values.shape[1]),importances[indices],color = 'lightblue',align = 'center')
plt.xticks(range(train_values.shape[1]), feature_labels, rotation = 90)
plt.xlim([-1,train_values.shape[1]])
plt.tight_layout()
plt.show()
