#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# We define a set of values
data = [['green', 'M', 10.1, 'class1'],['red', 'L', 13.5, 'class2'],['blue', 'XL', 15.3, 'class1']]
# Store the values in a data frame
df = pd.DataFrame(data)
# Define the data labels
df.columns = ['color','size','price','classlabel']
print(df,'\n')


# Before using the OneHotEncoder we need to convert strings to numerical values

# We map ordinal features
size_mapping = {'XL': 3,'L': 2,'M': 1}
df['size'] = df['size'].map(size_mapping)
print(df,'\n')

# We encode nominal features by whatever integer
values = df[['color', 'size', 'price']].values
color_encoder = LabelEncoder()
values[:,0] = color_encoder.fit_transform(values[:,0])
print(values,'\n')


# We specify the color feature rank, if we set sparse=false we get a numpy array
hot_encoder = OneHotEncoder(categorical_features = [0])
# transform returns a sparse matrix that we convert to dense numpy arrayfor visualization
value_array = hot_encoder.fit_transform(values).toarray()
print(value_array,'\n')

# get_dummpies directly converts string columns with one-hot encoding
print(pd.get_dummies(df[['price','color','size']]))

