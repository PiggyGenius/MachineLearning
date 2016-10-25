#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import pandas as pd

# We define a set of values
values = [['green', 'M', 10.1, 'class1'],['red', 'L', 13.5, 'class2'],['blue', 'XL', 15.3, 'class1']]
# Store the values in a data frame
df = pd.DataFrame(values)
# Define the data labels
df.columns = ['color','size','price','classlabel']
print(df,'\n')

# We define a mapping
size_mapping = {'XL': 3,'L': 2,'M': 1}
# And we apply it
df['size'] = df['size'].map(size_mapping)
print(df,'\n')

# Now we define an inverse on the items
inv_size_mapping = {v: k for k, v in size_mapping.items()}
# We apply it
df['size'] = df['size'].map(inv_size_mapping)
print(df)

