#!/usr/bin/python2.7
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
sns.set_style("whitegrid")


#####################################################################################################
# ************************************************************************************************* #
# *											TITANIC DATA										  * #
# ************************************************************************************************* #
# VARIABLE DESCRIPTIONS:																			#
# survival		Survival: (0 = No; 1 = Yes)															#
# pclass      	Passenger Class: (1 = 1st; 2 = 2nd; 3 = 3rd)										#
# name            Name																				#
# sex             Sex																				#
# age             Age																				#
# sibsp           Number of Siblings/Spouses Aboard													#
# parch           Number of Parents/Children Aboard													#
# ticket          Ticket Number																		#
# fare            Passenger Fare																	#
# cabin           Cabin																				#
# embarked        Port of Embarkation: (C = Cherbourg; Q = Queenstown; S = Southampton)				#
#																									#
# ************************************************************************************************* #
# *											SPECIAL INFO										  * #
# ************************************************************************************************* #
#																									#
# Pclass is a proxy for socio-economic status (SES):  1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower		#
# Age is in Years; Fractional if Age less than One (1). If Estimated, it is in the form xx.5		#
# Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic					#
# Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)			#
# Parent:   Mother or Father of Passenger Aboard Titanic											#
# Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic						#
#																									#
#####################################################################################################


def prints():
    print("---------------------------------------------------------------------------------------------------------")

# Understanding the data
titanic_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")
print(titanic_df.head());prints()
print(titanic_df.info());prints()
print(test_df.info())

# Removing useless features
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis = 1)
test_df = test_df.drop(['Name','Ticket'], axis = 1)

# We fill the two missing values of embarked feature with most common value
print(titanic_df['Embarked'].value_counts())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')

# We plot the ratio of survival based on the harbor of embarkation
sns.factorplot('Embarked', 'Survived', data = titanic_df, size = 6, aspect = 3)
sns.plt.title('Ratio of survival with respect to the harbor of embarkation')
sns.plt.show()

fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (22, 10))
# We plot the count of passengers that embarked from each harbor
sns.countplot(x = 'Embarked', data = titanic_df, ax = axis1)
axis1.set_title('Count of passengers that embarked from each harbor')
# We plot the count of passangers that survived with respect to their harbor of embarkation
sns.countplot(x = 'Survived', hue = "Embarked", data = titanic_df, order = [1,0], ax = axis2)
axis2.set_title('Count of passengers that survived with respect to their harbor of embarkation')

# We plot the mean for survived passengers for each value in Embarked
embark_perc = titanic_df[["Embarked", "Survived"]].groupby(['Embarked'],as_index = False).mean()
sns.barplot(x = 'Embarked', y = 'Survived', data = embark_perc, order = ['S', 'C', 'Q'], ax = axis3)
axis3.set_title('Mean of survived passengers with respect to their harbor')
sns.plt.show()

# Logically, Embarked doesn't seem to be useful in prediction.
titanic_df.drop(['Embarked'], axis = 1, inplace = True)
test_df.drop(['Embarked'], axis = 1, inplace = True)

# There is a missing 'Fare' value in test_df
test_df["Fare"].fillna(test_df["Fare"].median(), inplace = True)

titanic_df['Fare'] = titanic_df['Fare'].astype(int)
test_df['Fare'] = test_df['Fare'].astype(int)
fare_not_survived = titanic_df["Fare"][titanic_df["Survived"] == 0]
fare_survived = titanic_df["Fare"][titanic_df["Survived"] == 1]

# Wet get the average and the deviation of 'Fare' of survived/not survived passengers
avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])

# We plot the histogram to display the amount of tickets for each possible price
titanic_df['Fare'].plot(kind = 'hist', figsize = (15,5), bins = 100, xlim = (0,50))
sns.plt.title('Histogram of Fare prices')
sns.plt.show()
avgerage_fare.index.names = std_fare.index.names = ["Survived"]
avgerage_fare.plot(yerr = std_fare, kind = 'bar', legend = False)
sns.plt.show()
