# Perform bipartite spectral graph co-clustering.

"""
Based on the algorithm by Dhillon (2001) (dhillon2001co-clustering),
as introduced for dialect clustering by Wieling and Nerbonne (2009, 2010, 2011)
(wieling2009bipartite, wieling2010hierarchical, wieling2011bipartite).
"""
import math
import numpy as np
from scipy import linalg
from sklearn import cluster


def bsgc(A, k, doculects, all_correspondences):
    n_samples = len(doculects)
    n_features = len(all_correspondences)
    # Form the normalized matrix A_n.
    # Note that I already raise D_1, D_2 to the power of -0.5.
    D_1 = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        D_1[i, i] = np.sum(A[i])
    D_1 = linalg.sqrtm(np.linalg.inv(D_1))

    D_2 = np.zeros((n_features, n_features))
    col_sum = np.sum(A, axis=0)
    for j in range(n_features):
        D_2[j, j] = col_sum[j]
    D_2 = linalg.sqrtm(np.linalg.inv(D_2))

    A_n = D_1 @ A @ D_2

    # Get the singular vectors of A_n.
    U, S, V_T = np.linalg.svd(A_n)
    V = np.transpose(V_T)

    # Use the singular vectors to get the eigenvectors.
    n_eigenvecs = math.ceil(math.log(k, 2))

    Z = np.zeros((n_samples + n_features, n_eigenvecs))
    Z[:n_samples] = D_1 @ U[:, 1:n_eigenvecs + 1]
    Z[n_samples:] = D_2 @ V[:, 1:n_eigenvecs + 1]

    kmeans = cluster.KMeans(k)
    clusters = kmeans.fit_predict(Z)
    clusters_and_doculects = list(zip(clusters[:n_samples], doculects))
    clusters_and_features = list(zip(clusters[n_samples:],
                                     all_correspondences))
    return clusters_and_doculects, clusters_and_features


def split_features(A, doculects, all_correspondences,
                   clusters_and_doculects, clusters_and_features):
    rows = [[], []]
    docs = [[], []]
    for i, (c, d) in enumerate(clusters_and_doculects):
        rows[c].append(i)
        docs[c].append(d)
    cols = [[], []]
    corres = [[], []]
    for i, (c, f) in enumerate(clusters_and_features):
        cols[c].append(i)
        corres[c].append(f)

    # Occasionally, sound correspondences whose eigenvalues are close to the
    # kmeans decision boundary end up in the "wrong" cluster in that they do
    # not belong to any of the doculects in this cluster. If the doculects-to-
    # features matrix contains empty columns(/rows), it cannot be normalized
    # in bsgc(). To prevent this, we change the labels of such instances.
    for cur, other in ((0, 1), (1, 0)):
        if docs[cur] and corres[cur]:
            A_new = A[rows[cur]]
            A_new = A_new[:, cols[cur]]
            non0 = np.nonzero(A_new)
            rows_non0 = set(non0[0])
            rows_to_move = []
            docs_to_move = []
            for c in range(A_new.shape[0]):
                if c not in rows_non0:
                    print("!! Moving {} from {} to {}.".format(docs[cur][c],
                                                               docs[cur],
                                                               docs[other]))
                    rows_to_move.append(rows[cur][c])
                    docs_to_move.append(docs[cur][c])
            cols_non0 = set(non0[1])
            cols_to_move = []
            corres_to_move = []
            for c in range(A_new.shape[1]):
                if c not in cols_non0:
                    print("\tMoving {} from {} to {}.".format(corres[cur][c],
                                                              docs[cur],
                                                              docs[other]))
                    cols_to_move.append(cols[cur][c])
                    corres_to_move.append(corres[cur][c])

            for items, items_to_move in ((rows, rows_to_move),
                                         (docs, docs_to_move),
                                         (cols, cols_to_move),
                                         (corres, corres_to_move)):
                for i in items_to_move:
                    items[cur].remove(i)
                    items[other].append(i)

    A_0 = A[rows[0]]
    A_0 = A_0[:, cols[0]]
    A_1 = A[rows[1]]
    A_1 = A_1[:, cols[1]]
    return (A_0, tuple(docs[0]), corres[0]), (A_1, tuple(docs[1]), corres[1])


def bsgc_recursive(A, doculects, all_correspondences, clusters=None):
    clusters_and_doculects, clusters_and_features = bsgc(A, 2, doculects,
                                                         all_correspondences)
    new_features = split_features(A, doculects, all_correspondences,
                                  clusters_and_doculects,
                                  clusters_and_features)
    if clusters is None:
        clusters = {tuple(doculects): all_correspondences}
    for (A, docs, corres) in new_features:
        clusters[docs] = corres
        if len(docs) > 1:
            clusters = bsgc_recursive(A, docs, corres, clusters)
    return clusters


def bsgc_hierarchical(A, doculects, all_correspondences):
    clusters = bsgc_recursive(A, doculects, all_correspondences)
    clusters = sorted(clusters.items(), key=lambda x: len(x[0]))
    clusters = [c for c in clusters if len(c[0]) > 0]
    return [c[0] for c in clusters], [c[1] for c in clusters]
