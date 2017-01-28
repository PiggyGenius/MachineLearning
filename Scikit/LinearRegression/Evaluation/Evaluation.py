#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

####################################################################################################
# ************************************************************************************************ #
# *                                      Regression Models                                         #
# ************************************************************************************************ #
#                                                                                                  #
# We can use different regression models:                                                          #
#   - Ridge:                                                                                       #
#       - from sklearn.linear_model import Ridge                                                   #
#       - ridge = Ridge(alpha = 1.0).                                                              #
#   - LASSO:                                                                                       #
#       - from sklearn.linear_model import LASSO                                                   #
#       - lasso = Lasso(alpha = 1.0).                                                              #
#   - Elastic Net:                                                                                 #
#       - from sklearn.linear_model import ElasticNet                                              #
#       - elastic_net = ElasticNet(alpha = 1.0, l1_ratio = 0.5).                                   #
#                                                                                                  #
####################################################################################################

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

# We load the housing dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = df.iloc[:, :-1].values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# We plot a residual plot where we substract the true target variables from our predicted responses
plt.scatter(y_train_pred, y_train_pred - y_train, c = 'blue', marker = 'o', label = 'Training data')
plt.scatter(y_test_pred, y_test_pred - y_test, c = 'lightgreen', marker = 's', label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = -10, xmax = 50, lw = 2, color = 'red')
plt.xlim([-10, 50])
plt.show()

# The Mean Squared Error is a quantitative measure of a model's performance
print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train,y_train_pred), mean_squared_error(y_test, y_test_pred)))
print('The MSE on the training set is much lower than the one on the test set which is an indicator that our model is overfitting.')

# We compute the coefficient of determination R^2
print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))
