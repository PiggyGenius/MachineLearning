#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
                                        
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

# We set the maximum number of iterations to 100 and we use at least 50 samples for each trial
# We set the residual_threshold parameter to 5.0, samples are included in the inlier set if their vertical distance to the fitted line is within 5 distance units
# Scikit uses the Median Absolute Deviation of the target values yto select the inlier threshold
# The choise of an appropriate value for the inlier threshold is problem-specific
ransac = RANSACRegressor(LinearRegression(), max_trials = 100, min_samples = 50, loss = 'absolute_loss', residual_threshold = 5.0, random_state = 0)
ransac.fit(X, y)

# We obtain the inliers and outliers from the fitted RANSAC, plot them together with the linear fit
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c = 'blue', marker = 'o', label = 'Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c = 'lightgreen', marker = 's', label = 'Outliers')
plt.plot(line_X, line_y_ransac, color = 'red')
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000\'s [MEDV]')
plt.legend(loc = 'upper left')
plt.show()

print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)
