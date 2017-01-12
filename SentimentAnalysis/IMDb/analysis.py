#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('./movie_data.csv')
print(df.head(3))

# We create the bag-of-words and print vocabulary as well as feature vectors
count = CountVectorizer()
docs = np.array(['The sun is shining', 'The weather is sweet', 'The sun is shining and the teacher is sweet'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())
