#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import pandas as pd
from sklearn.preprocessing import Imputer
from io import StringIO

# In CSV you seperate columns and values by commas, each \n is a new row
csv_data = "A,B,C,D\n1.0,2.0,3.0,4.0\n5.0,6.0,,8.0\n0.0,11.0,12.0,"

# StringIO allows us to use csv_data string as if it was a file, in-memory buffer
# We read the csv and the missing values are automatically changed to NaN
df = pd.read_csv(StringIO(csv_data))
print(df,'\n')

# We replace NaN values by the mean value of the column, axis=0, or row with axis=1
# strategy could be: 'median', 'most_frequent'
# replacing NaN by most_frequent value is useful for imputing categorical values
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)

# We fit the Imputer on our DataFrame
imr = imr.fit(df)

# Now we just have to apply the Imputer
imputed_data = imr.transform(df.values)
print(imputed_data)
