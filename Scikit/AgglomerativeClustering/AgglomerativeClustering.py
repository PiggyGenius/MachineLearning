#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# We generate random sample data to work with: 4 samples and 3 features
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns = variables, index = labels)

ac = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)
print('\n', 'ID_0, ID_3 and ID_4 are assigned to cluster 0 ; ID_1 and ID_2 are assigned to cluster 1')
print('Results are consistent with the dendrogen constructed in NoLibraries/HierarchicalClustering')
