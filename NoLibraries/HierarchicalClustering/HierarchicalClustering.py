#!/usr/bin/python3.5
# -*-coding:Utf-8 -*
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram

# We generate random sample data to work with: 4 samples and 3 features
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns = variables, index = labels)
print(df)

# We calculate the distance matrix as input for the hierarchical clustering algorithm
# We calculate the Euclidean distance between each pair of sample points based on our features
# We provide the condensed distance matrix as input to the squareform function to create a symmetrical matrix on the pair-wise distances
row_dist = pd.DataFrame(squareform(pdist(df, metric = 'euclidean')), columns = labels, index = labels)
print('\n', row_dist)

# We apply the complete linkage agglomeration to our clusters and get a linkage matrix
# We can use a condensed distance matrix, upper triangular from the pdist function as an input
# We can also provide the initial data array and use the Euclidean metric as argument in linkage
# We shouldn't use the squareform distance matrix defined earlier since it would yield different distance values from those expected
# INCORRECT: row_clusters = linkage(row_dist, method = 'complete', metric = 'euclidean')
# CORRECT: row_clusters = linkage(pdist(df, metric = 'euclidean'), method = 'complete')
row_clusters = linkage(df.values, method = 'complete', metric = 'euclidean')

# We can turn the clustering results into Pandas DataFrame
columns = ['row label 1', 'row label 2', 'distance', 'number of items in cluster']
index = ['cluster %d' %(i + 1) for i in range(row_clusters.shape[0])]
print('\n',pd.DataFrame(row_clusters, columns = columns, index = index))

# Now that we have computed the linkage matrix, we can visualize the results in form of a dendrogram
row_dendr = dendrogram(row_clusters, labels = labels)
plt.tight_layout()
plt.ylabel('Euclidean distance')
plt.show()
print('\n', 'Such a dendrogram summarizes the different clusters that were formed during the agglomerative hierarchical clustering. We can see that the samples ID_0 and ID_4, followed by ID_1 and ID_2, are the most similar ones based on the Euclidean distance metric.')

######################################################################################################
#                                                                                                    #
#                       WE WANT TO ATTACH THE DENDROGRAM TO A HEAT MAP                               #
#                                                                                                    #
######################################################################################################

# We create a figure object and define the x and y axis poistions, width and height of the dendrogram
# We rotate the dendrogram 90 degree counter-clockwise
fig = plt.figure(figsize = (8, 8))
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation = 'right')

# We reorder data in our DataFrame according to clustering labels
df_rowclust = df.ix[row_dendr['leaves'][::-1]]

# We construct the heat map and put it next to the dendrgoram
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation = 'nearest', cmap = 'hot_r')

# We remove the axis ticks, the axis spines, add a color bar, assign feature and sample names to x, y
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()
