#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pyprind
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

stop = stopwords.words('english')

# Cleans the unprocessed text data, seperates it into word tokens and removes stop words
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# Generator function that reads in and return one document at a time
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# Takes a document stream from the stream_docs and return size documents 
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

# We can't use the CountVectorizer since it requires holding the complete vocabulary in memory
# The TfidfVectorizer keeps feature vectors in memory to calculate the inverse document frequencies
# HashingVectorizer is data-independent and makes use of the Hashing trick via MurmurHash3 algorithm
# We set the number of features to 2^21, reduces the chance to cause hash collisions
vect = HashingVectorizer(decode_error = 'ignore', n_features = 2**21, preprocessor = None, tokenizer = tokenizer)
clf = SGDClassifier(loss = 'log', random_state = 1, n_iter = 1)
doc_stream = stream_docs(path = './movie_data.csv')

# We make use of the partial_fit function of the SGDClassifier to stream the documents directly from our local drive and train a logistic regression model using small minibatches of documents
# 45 minibatches, each minibatch consists of 1000 documents, we use the last 5000 for evaluation
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    train_values, train_classes = get_minibatch(doc_stream, size = 1000)
    if not train_values:
        break
    train_values = vect.transform(train_values)
    clf.partial_fit(train_values, train_classes, classes = classes)
    pbar.update()

test_values, test_classes = get_minibatch(doc_stream, size = 5000)
test_values = vect.transform(test_values)
print('Accuracy: {:.3f}'.format(clf.score(test_values, test_classes)))
