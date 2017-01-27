#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from LinearRegressionGD import LinearRegressionGD

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

X = df[['RM']].values
y = df['MEDV'].values
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

lr = LinearRegressionGD()
lr.fit(X_std, y_std)

# We plot the cost as a function of the number of epochs to check for convergence
plt.plot(range(1, lr.n_iter + 1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()
print('We can see convergence after the fith epoch')

# Plots a scatterplot of the training samples and add the regression line
def lin_regplot(X, y, model):
    plt.scatter(X, y, c = 'blue')
    plt.plot(X, model.predict(X), color = 'red')
    return None

# We plot the number of rooms against house prices
lin_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM] (standardized)')
plt.ylabel('Price in $1000\'s [MEDV] (standardized)')
plt.show()

# To scale the predicted outcome back on the Price in $1000's axes, we inverse_transform the scaler
num_rooms_std = sc_x.transform(np.array([[5.0]]))
price_std = lr.predict(num_rooms_std)
# We don't have to update the weights of the intercept if working with standardized variables.
# The y axis intercept is always 0.
print('Slope: %.3f' % lr.w_[1])
print('Intercept: %.3f' % lr.w_[0])
print('\nWe use our model to predict the price of a house with five rooms.')
print('Price in $1000\'s: %.3f' % sc_y.inverse_transform(price_std))
