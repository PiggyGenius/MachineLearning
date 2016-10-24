#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import pandas as pd
from io import StringIO

# In CSV you seperate columns and values by commas, each \n is a new row
csv_data = "A,B,C,D\n1.0,2.0,3.0,4.0\n5.0,6.0,,8.0\n0.0,11.0,12.0,"

# StringIO allows us to use csv_data string as if it was a file, in-memory buffer
# We read the csv and the missing values are automatically changed to NaN
df = pd.read_csv(StringIO(csv_data))
print(df,'\n')

# isnull returns true or false if value is NaN or not
# We print the sum of null values for each column
print(df.isnull().sum(),'\n')

# Drop rows containing at least 1 NaN
print(df.dropna(),'\n')

# drop columns containing at least 1 NaN
print(df.dropna(axis=1),'\n')

# only drop rows where all columns are NaN
print(df.dropna(how='all'),'\n')

# drop rows that have not at least 4 non-NaN values
print(df.dropna(thresh=4),'\n')

# only drop rows where NaN appear in specific columns (here: 'C')
print(df.dropna(subset=['C']))


# df.values returns the underlying numpy array of the DataFrame
