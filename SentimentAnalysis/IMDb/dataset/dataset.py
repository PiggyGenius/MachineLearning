#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pyprind
import pandas as pd
import os

#############################################################################################
#                                                                                           #
# The script was used on the dataset of imdb reviews available on stanford/data/sentiment.  #
# Download it to run the script.                                                            #
#                                                                                           #
#############################################################################################

# We read the files and append them to the dataframe, we create a progress bar for visualization
pbar = pyprind.ProgBar(50000)
labels = {'pos':1, 'neg':0}
df = pd.DataFrame();
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = './{}/{}'.format(s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index = True)
            pbar.update()
df.columns = ['review', 'sentiment']

# We shuffle the dataframe and store the result in a csv file, reread it to check
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('../movie_data.csv', index = False)
