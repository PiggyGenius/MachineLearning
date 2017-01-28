#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

####################################################################################################
# ************************************************************************************************ #
# *                                       HOUSING DATASET                                          #
# ************************************************************************************************ #
#                                                                                                  #
# The features of the 506 samples can be summarized as:                                            #
#   - CRIM: Per capita crime rate by town.                                                         #
#   - ZN: Proportion of residential land zoned for lots larger than 25000 sq.ft.                   #
#   - INDUS: Proportion of non-retail business acres per town.                                     #
#   - CHAS: Charles River dummy variable, equal to 1 if tract bounds river and 0 otherwise.        #
#   - NOX: Nitric oxides concentration, parts per 10 million.                                      #
#   - RM: Average number of rooms per dwelling.                                                    #
#   - AGE: Proportion of owner-occupied units built prior to 1940.                                 #
#   - DIS: Weighted distances to five Boston employment centers.                                   #
#   - RAD: Index of accessibility to radial highways.                                              #
#   - TAX: Full-value property-tax rate per $10000.                                                #
#   - PTRATIO: Pupil-teacher ratio by town.                                                        #
#   - B: Calculated as 1000(Bk - 0.63)^2, Bk: proportion of people of African descent by town.     #
#   - LSTAT: Percentage lower status of the population.                                            #
#   - MEDV: Median value of owner-occupied homes in $1000s.                                        #
#                                                                                                  #
####################################################################################################

# Plots a scatterplot of the training samples and add the regression line
def lin_regplot(X, y, model):
    plt.scatter(X, y, c = 'blue')
    plt.plot(X, model.predict(X), color = 'red')
    return None

# We load the housing dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# We model the nonlinear relationship between the MEDV and LSTAT variables
X = df[['LSTAT']].values
y = df['MEDV'].values
tree = DecisionTreeRegressor(max_depth = 3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.show()
print('This model does not capture the continuity and differentiability of the desired prediction.')
print('Chose the approriate value for the depth of the tree to not overfit or underfit the data.')
