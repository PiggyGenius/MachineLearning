#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

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

# We model the relationship between house prices and LSTAT (percent lower status of the population)
# To do so, we use quadratic and cubic polynomials and compare it to a linear fit

# We load the housing dataset
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data', header = None, sep = '\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

# We pick the two features on which we want to model the linear regression
X = df[['LSTAT']].values
y = df['MEDV'].values
regression = LinearRegression()

# Create polynomial features
quadratic = PolynomialFeatures(degree = 2)
cubic = PolynomialFeatures(degree = 3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

# Linear fit
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regression = regression.fit(X, y)
y_lin_fit = regression.predict(X_fit)
linear_r2 = r2_score(y, regression.predict(X))

# Quadratic fit
regression = regression.fit(X_quad, y)
y_quad_fit = regression.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regression.predict(X_quad))

# Cubic fit
regression = regression.fit(X_cubic, y)
y_cubic_fit = regression.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regression.predict(X_cubic))

# Plot results
plt.scatter(X, y, label='training points', color='lightgray')
plt.plot(X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2, color='blue', lw=2, linestyle=':')
plt.plot(X_fit, y_quad_fit, label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2, color='red', lw=2, linestyle='-')
plt.plot(X_fit, y_cubic_fit, label='cubic (d=3), $R^2=%.2f$' % cubic_r2, color='green', lw=2, linestyle='--')
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc = 'upper right')
plt.show()

# The cubic fit captures the relationship between the house prices and LSTAT better than the linear and quadratic fit. 
# Adding more and more polynomial features increases the complexity of a model and therefore increases the chance of overfitting. 
# It is always recommended that you evaluate the performance of the model on a separate test dataset to estimate the generalization performance. 
# In addition, polynomial features are not always the best choice for modeling nonlinear relationships.

# Just by looking at the MEDV-LSTAT scatterplot, we could propose that a log transformation of the LSTAT feature variable and the square root of MEDV may project the data onto a linear feature space suitable for a linear regression fit.

# We fit features
X_log = np.log(X)
y_sqrt = np.sqrt(y)
X_fit = np.arange(X_log.min() - 1, X_log.max() + 1, 1)[:, np.newaxis]
regression = regression.fit(X_log, y_sqrt)
y_lin_fit = regression.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regression.predict(X_log))

# We plot the results
plt.scatter(X_log, y_sqrt, label = 'training points', color = 'lightgray')
plt.plot(X_fit, y_lin_fit, label = 'linear (d=1), $R^2=%.2f$' % linear_r2, color = 'blue', lw = 2)
plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000\'s [MEDV]}$')
plt.legend(loc = 'lower left')
plt.show()
