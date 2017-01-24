#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

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

# We load the movie review data
df = pd.read_csv('./movie_data.csv')
print(df.head(3))

# We clean the text data to remove punctuation, we keep emoticons
def preprocessor(text):
    # We remove the entire HTML markup contained in the movie reviews
    text = re.sub('<[^>]*>', '', text)
    # We store empticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # We remove all non words characters, add emoticons at the end or the string and remove nose '-'
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text
print(preprocessor(df.loc[0, 'review'][-50:]))
print(preprocessor("</a>This :) is :( a test :-)!"))

# We can split our data using tokens for each word
def tokenizer(text):
    return text.split()
print(tokenizer('runners like running and thus they run'))

# A better process is to use word stemming to transform a word into its root form
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in  text.split()]
print(tokenizer_porter('runners like running and thus they run'))

# To remove stop-words from the movie reviews we use the nltk set of stop-words
nltk.download('stopwords')
stop = stopwords.words('english')
print([w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:] if w not in stop])
