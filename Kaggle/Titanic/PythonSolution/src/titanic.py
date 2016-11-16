#!/usr/bin/python3.5
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

def prints():
    print("---------------------------------------------------------------------------------------------------------")

# Understanding the data
titanic_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
print(titanic_df.head());prints()
print(titanic_df.info());prints()
print(test_df.info())

# Removing useless features
titanic_df = titanic_df.drop(['PassengerId','Name','Ticket'], axis = 1)
test_df = test_df.drop(['Name','Ticket'], axis = 1)

# We fill the two missing values of embarked feature with most common value
print(titanic_df['Embarked'].value_counts())
titanic_df['Embarked'] = titanic_df['Embarked'].fillna('S')
sns.factorplot('Embarked','Survived');
