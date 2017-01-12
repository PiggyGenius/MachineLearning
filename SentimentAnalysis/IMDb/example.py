#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# We create the bag-of-words and print vocabulary as well as feature vectors
count = CountVectorizer()
docs = np.array(['The sun is shining', 'The weather is sweet', 'The sun is shining and the teacher is sweet'])
bag = count.fit_transform(docs)
print('Vocabulary:',count.vocabulary_)
print('\nEach index position resembles the count of the word at this index in the vocabulary.')
print(bag.toarray(),'\n')

# The TFIDF transformer takes raw term frequencies vector from CountVectorizer and transforms them into tf-idfs
tfidf = TfidfTransformer()
np.set_printoptions(precision = 2)
print('TF-IDF vectors: ')
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
