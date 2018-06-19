# Reproducing the example from Wieling and Nerbonne (2011):
# "Bipartite spectral graph partitioning for clustering dialect varieties and
# detecting their linguistic features", which is based on
# Dhillon (2001): "Co-clustering documents and words using Bipartite Spectral
# Graph Partitioning"

import math
import numpy as np
import scipy.linalg
import sklearn.cluster
from numpy import linalg as la

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

doculects = ['Appelscha (Friesland)', 'Oudega (Friesland)', 'Vaals (Limburg)',
             'Sittard (Limburg)']
correspondences = ['ʌ:ɪ', '-:ə', 'd:w']
n_samples = len(doculects)
n_features = len(correspondences)

A = np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1]])

# 1) Form the normalized matrix A_n.

D_1 = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    D_1[i, i] = np.sum(A[i])
D_1 = scipy.linalg.sqrtm(la.inv(D_1))
print("D_1^-0.5")
print(D_1)

D_2 = np.zeros((n_features, n_features))
for j in range(n_features):
    D_2[j, j] = np.sum(A, axis=0)[j]
D_2 = scipy.linalg.sqrtm(la.inv(D_2))
print("D_2^-0.5")
print(D_2)

A_n = D_1 @ A @ D_2
print("A_n")
print(A_n)

# 2) Get the singular values and vectors of A_n.

U, S, V_T = la.svd(A_n)
V = np.transpose(V_T)

print("U, S, V^T")
print(U)
print(S)
print(V_T)

# 3) Use the singular vectors to get the eigenvectors.

# number of clusters
# k = 2
k = 3

# Rounding (up) isn't mentioned, but it seems necessary to me to get proper
# slices.)
l = math.ceil(math.log(k, 2))
# l = int(round(math.log(k, 2)))
print(l)

# TODO Why is this different from the actual U created in this program?
U = np.array([[-.5, .5, .71, 0],
              [-.5, .5, -.71, 0],
              [-.5, -.5, 0, -.71],
              [-.5, -.5, 0, .71]])
V = np.array([[-.5, .71, -.5],
              [-.71, 0, .71],
              [-.5, -.71, -.5]])
D_1 = np.array([[.71, 0, 0, 0],
                [0, .71, 0, 0],
                [0, 0, .71, 0],
                [0, 0, 0, .71]])
D_2 = np.array([[.71, 0, 0],
                [0, .5, 0],
                [0, 0, .71]])


Z = np.zeros((n_samples + n_features, l))
# Does the l+1 in the paper take care of rounding up? (Which we did above.)
# Why are we ignoring the first column/row?
Z[:n_samples] = D_1 @ U[:, 1:1 + l]
Z[n_samples:] = D_2 @ V[:, 1:1 + l]

print(Z)

# 4) Run k-means on Z.

kmeans = sklearn.cluster.KMeans(k)
clusters = kmeans.fit_predict(Z)
print(clusters)

for i, d in enumerate(clusters[:n_samples]):
    print("{} belongs to cluster {}".format(doculects[i], d))
for i, d in enumerate(clusters[n_samples:]):
    print("{} describes cluster {}".format(correspondences[i], d))
