#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN

# We create two half moons of 100 samples each
X, y = make_moons(n_samples = 200, noise = 0.05, random_state = 0)
plt.scatter(X[:, 0], X[:, 1])
plt.show()

# We use K-means clustering to see if they can be successful on this data
f, (ax1, ax2) = plt.subplots(1, 2,)
km = KMeans(n_clusters = 2, random_state = 0)
y_km = km.fit_predict(X)
ax1.scatter(X[y_km == 0, 0], X[y_km == 0, 1], c = 'lightblue', marker='o', s = 40, label = 'cluster 1')
ax1.scatter(X[y_km == 1, 0], X[y_km == 1, 1], c = 'red', marker = 's', s = 40, label = 'cluster 2')
ax1.set_title('K-means clustering')

# We use complete linkage clustering to see if they can be successful on this data
ac = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'complete')
y_ac = ac.fit_predict(X)
ax2.scatter(X[y_ac == 0, 0], X[y_ac == 0, 1], c = 'lightblue', marker = 'o', s = 40, label = 'cluster 1')
ax2.scatter(X[y_ac == 1, 0], X[y_ac == 1, 1], c = 'red', marker = 's', s = 40, label = 'cluster 2')
ax2.set_title('Agglomerative clustering')

# We plot the results
plt.legend()
plt.tight_layout()
plt.show()

# Since the other methods failed, we use DBSCAN
db = DBSCAN(eps = 0.2, min_samples = 5, metric = 'euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db == 0, 0], X[y_db == 0, 1], c = 'lightblue', marker = 'o', s = 40, label = 'cluster 1')
plt.scatter(X[y_db == 1, 0], X[y_db == 1, 1], c = 'red', marker = 's', s = 40, label = 'cluster 2')
plt.legend()
plt.tight_layout()
plt.show()
