#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from SBS import SBS

# Retreiving data
wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
df_wine = pd.read_csv(wine_url,header=None)
columns = ["Class label","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
df_wine.columns = columns 

# Splitting data
classes =  df_wine.iloc[:, 0].values
values =  df_wine.iloc[:, 1:].values
train_values, test_values, train_classes, test_classes = train_test_split(values, classes, test_size = 0.3, random_state = 0)

# Scaling data
stdsc  = StandardScaler()
std_train_values = stdsc.fit_transform(train_values)
std_test_values = stdsc.transform(test_values)

# Fitting data wih KNN and SBS
knn = KNeighborsClassifier(n_neighbors = 2)
sbs = SBS(knn, k_features = 1)
sbs.fit(std_train_values,train_classes)
k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5],'\n')

# Fitting on original data
knn.fit(std_train_values,train_classes)
print("Original data:")
print("Training accuracy: ", knn.score(std_train_values, train_classes))
print("Test accuracy: ", knn.score(std_test_values, test_classes),'\n')

# Fitting on the subset given by SBS
knn.fit(std_train_values[:, k5], train_classes)
print("SBS data:")
print('Training accuracy:',knn.score(std_train_values[:, k5], train_classes))
print('Test accuracy:',knn.score(std_test_values[:, k5],test_classes))

# Plotting data
k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()



##############            Notes about the figure              ###############
#                                                                           #
# The accuracy of the KNN classifier improved on the validation dataset.    #
# It is likely due to the curse of dimensionality.                          #
# The accuracy is a bit smaller on the test data --> overfitting.
#                                                                           #
##############            Notes about the figure              ###############
