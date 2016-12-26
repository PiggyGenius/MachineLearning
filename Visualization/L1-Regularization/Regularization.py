#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


# Displaying weigths with different regularization strengths
fig = plt.figure()
ax = plt.subplot(111)
colors = ["blue", "green", "red", "cyan","magenta", "yellow", "black","pink", "lightgreen", "lightblue","gray", "indigo", "orange"]
weights, params = [], []

for c in np.arange(-4, 6):
    logistic_regression = LogisticRegression(penalty = "l1",C = 10**c,random_state = 0)
    logistic_regression.fit(std_train_values, train_classes)
    weights.append(logistic_regression.coef_[1])
    params.append(10**c)

weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params,weights[:,column],label = df_wine.columns[column+1],color = color)

plt.axhline(0,color = 'black',linestyle = '--',linewidth = 3)
plt.xlim([10**(-5),10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center',bbox_to_anchor = (1.38, 1.03),ncol = 1,fancybox = True)
plt.show()


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
