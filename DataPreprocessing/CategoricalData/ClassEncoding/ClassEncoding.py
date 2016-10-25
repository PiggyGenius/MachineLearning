#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# We define a set of values
values = [['green', 'M', 10.1, 'class1'],['red', 'L', 13.5, 'class2'],['blue', 'XL', 15.3, 'class1']]
# Store the values in a data frame
df = pd.DataFrame(values)
# Define the data labels
df.columns = ['color','size','price','classlabel']
print(df,'\n')

# We create a class mapping on the dataset
class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
print(class_mapping,'\n')
# We apply the mapping
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df,'\n')

# Same thing than before for the inversion
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df,'\n')

# We can use the scikit label encoder
label_encoder = LabelEncoder()
# fit transform calls fit then transform
classes = label_encoder.fit_transform(df['classlabel'].values)
print(classes,'\n')

# We can still invert data
print(label_encoder.inverse_transform(classes))
