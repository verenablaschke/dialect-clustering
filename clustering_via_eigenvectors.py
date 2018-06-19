# Reproducing the example from Wieling and Nerbonne (2011):
# Bipartite spectral graph partitioning for clustering dialect varieties and
# detecting their linguistic features

import math
import numpy as np
import scipy.linalg
import sklearn.cluster
from numpy import linalg as la

doculects = ['Appelscha (Friesland)', 'Oudega (Friesland)', 'Vaals (Limburg)',
             'Sittard (Limburg)']
correspondences = ['ʌ:ɪ', '-:ə', 'd:w']
n_samples = len(doculects)
n_features = len(correspondences)

A = np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1]])

# 1) Form the normalized matrix A_n.

D_1 = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    D_1[i][i] = np.sum(A[i])
D_1 = scipy.linalg.sqrtm(la.inv(D_1))
print("D_1^-0.5")
print(D_1)

D_2 = np.zeros((n_features, n_features))
for j in range(n_features):
    D_2[j][j] = np.sum(A, axis=0)[j]
D_2 = scipy.linalg.sqrtm(la.inv(D_2))
print("D_2^-0.5")
print(D_2)

A_n = D_1 @ A @ D_2
print("A_n")
print(A_n)

# 2) Get the singular values and vectors of A_n.

# It seems like in the example they truncated A to 2 floating point digits,
# like so: (This yields somewhat different results, although the divergent
# results here might also be caused by general floating point rounding
# issues.)
# A_n = np.array([[.5, .35, 0], [.5, .35, 0], [0, .35, .5], [0, .35, .5]])

U, S, V_T = la.svd(A_n)
V = np.transpose(V_T)

print("U, S, V^T")
print(U)
print(S)
print(V_T)
print(V)

# 3) Use the singular vectors to get the eigenvectors.

# number of clusters
k = 2

# How does this work if l isn't an integer?
l = int(math.log(k, 2))
print(l)

# Again, the truncation of the floating point digits...
U[:][2] = [.5, .5, -.5, -.5]
V[2] = [.71, 0, -.71]
D_1 = np.array([[.71, 0, 0, 0], [0, .71, 0, 0], [0, 0, .71, 0], [0, 0, 0, .71]])
D_2 = np.array([[.71, 0, 0], [0, .5, 0], [0, 0, .71]])

# This isn't adapted yet to l > 1.
Z = np.zeros(n_samples + n_features)
Z[:n_samples] = D_1 @ U[:][2]
Z[n_samples:] = D_2 @ V[2]

print(Z)

# 4) Run k-means on Z.

# Again, the truncation of the floating point digits...

kmeans = sklearn.cluster.KMeans(k)
clusters = kmeans.fit_predict(Z.reshape(-1, 1))
print(clusters)

for i, d in enumerate(clusters[:n_samples]):
    print("{} belongs to cluster {}".format(doculects[i], d))
for i, d in enumerate(clusters[n_samples:]):
    print("{} describes cluster {}".format(correspondences[i], d))
