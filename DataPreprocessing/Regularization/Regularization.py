#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


# Recovering wine data
wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
df_wine = pd.read_csv(wine_url,header=None)
columns = ["Class label","Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium","Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins","Color intensity","Hue","OD280/OD315 of diluted wines","Proline"]
df_wine.columns = columns 


# Splitting the dataset
classes =  df_wine.iloc[:, 0].values
values =  df_wine.iloc[:, 1:].values
train_values, test_values, train_classes, test_classes = train_test_split(values,classes,test_size=0.3,random_state=0)


# Standardizing data
stdsc  = StandardScaler()
std_train_values = stdsc.fit_transform(train_values)
std_test_values = stdsc.transform(test_values)


logistic_regression = LogisticRegression(penalty = "l1", C = 0.1)
logistic_regression.fit(std_train_values, train_classes)
print("Training accuracy: ", logistic_regression.score(std_train_values, train_classes))
print("Test accuracy: ", logistic_regression.score(std_test_values, test_classes))
print("Intercept terms: ", logistic_regression.intercept_)
print("Weight coefficients: ", logistic_regression.coef_)



############ Notes about intercept terms and weight coefficients ############
#                                                                           #
# We have multiclass dataset which means logistic regression uses OvR:      #
#   - First intercept: fits class 1 vs class 2 and 3                        #
#   - Second intercept: fits class 2 vs class 1 and 3                       #
#   - third intercept: fits class 3 vs class 1 and 2                        #
#                                                                           #
# The weight array contains three rows, one for each class.                 #
# The weight vectors are spars ---> few non-zero entries.                   #
# Because of L1 regularization, the model is robust to irrelevant features. #
#                                                                           #
############ Notes about intercept terms and weight coefficients ############
