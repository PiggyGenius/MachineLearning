#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import pandas as pd
import numpy as np
import nltk, re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

####################################################################################################
# ************************************************************************************************ #
# *                                 PYTHON REGULAR EXPRESSSIONS                                    #
# ************************************************************************************************ #
#                                                                                                  #
#   - '[]': used to indicate a set of characters:                                                  #
#       - [amk]: matches a, m or k.                                                                #
#       - [0-5][0-9]: matches all two digits numbers from 00 to 59.                                #
#       - [^5]: matches any character except 5.                                                    #
#   - 'A | B': creates a regular expression  that will match either A or B.                        #
#          Once A matches, B will not be testes further, this operator is never greedy.            #
#   - '.': matches any character except a newline.                                                 #
#   - 'ab?': matches 0 or 1 repetitions, wil match either 'a' or 'ab'.                             #
#   - 'ab*': matches as many repetitions as possible, 'a', 'ab', 'abb', 'abbb'...                  #
#   - '\W': matches any non_alphanumeric characters, equivalent to [^a-zA-Z0-9_].                  #
#                                                                                                  #
####################################################################################################

# Function to clean the text data, remove punctuation, we keep emoticons
def preprocessor(text):
    # We remove the entire HTML markup contained in the movie reviews
    text = re.sub('<[^>]*>', '', text)
    # We store empticons
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    # We remove all non words characters, add emoticons at the end or the string and remove nose '-'
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

# We can split our data using tokens for each word
def tokenizer(text):
    return text.split()

# A better process is to use word stemming to transform a word into its root form
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in  text.split()]

# We get the stopwords
#nltk.download('stopwords')
stop = stopwords.words('english')


# We load the movie review data
df = pd.read_csv('./movie_data.csv')

# We apply our preprocessor function to the dataset
df['review'] = df['review'].apply(preprocessor)

# We split the movie reviews into test and train sets
train_values = df.loc[:25000, 'review'].values
train_classes = df.loc[:25000, 'sentiment'].values
test_values = df.loc[25000:, 'review'].values
test_classes = df.loc[25000:, 'sentiment'].values

# # We use grid-search to find the optimal set of parameters
# # TfidfVectorizer combines CountVectorizer and TfidfTransformer
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
param_grid = [{ 'vect__ngram_range': [(1,1)],
		'vect__stop_words': [stop, None],
		'vect__tokenizer': [tokenizer, tokenizer_porter],
		'clf__penalty': ['l1', 'l2'],
		'clf__C': [1.0, 10.0, 100.0]},
		{'vect__ngram_range': [(1,1)],
		'vect__stop_words': [stop, None],
		'vect__tokenizer': [tokenizer, tokenizer_porter],
		'vect__use_idf':[False],
		'vect__norm':[None],
		'clf__penalty': ['l1', 'l2'],
		'clf__C': [1.0, 10.0, 100.0]}]
lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state = 0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid, scoring = 'accuracy',cv = 5, verbose = 1, n_jobs = -1)
gs_lr_tfidf.fit(train_values, train_classes)
print('Best parameter set: {}'.format(gs_lr_tfidf.best_params_))
print('CV accuracy: {:.3f}'.format(gs_lr_tfidf.best_score_))
clf = gs_lr_tfidf.best_estimator_
print('Test accuracy: {:.3f}'.format(clf.score(train_values,test_values)))
