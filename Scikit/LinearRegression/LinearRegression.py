#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
                                        
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
print(df.head())


# We create a scatterplot matrix, allows us to visualize pair-wise correlations between features
sns.set(style = 'whitegrid', context = 'notebook')
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size = 2.5)
plt.show()
sns.reset_orig()

# We plot the correlation matrix array as a heat map
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1.5)
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 15}, yticklabels = cols, xticklabels = cols)
plt.show()
print('Our target variable MEDV shows the largest correlation with the LSTAT variable (-0.74).There is a clear nonlinear relationship between LSTAT and MEDV. The correlation between RM and MEDV is also relatively high (0.70) and given the linear relationship between those two variables that we observed in the scatterplot, RM seems to be a good choice for an exploratory variable to introduce the concepts of a simple linear regression model.')
