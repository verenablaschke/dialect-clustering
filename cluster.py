from align import align
from scipy import linalg
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import sys


def construct_A(all_correspondences, correspondences, doculects,
                min_count=0, binary=False):
    corres2int = {x: i for i, x in enumerate(all_correspondences)}
    n_samples = len(doculects)
    n_features = len(all_correspondences)
    A = np.zeros((n_samples, n_features), dtype=np.int16)
    for i, doculect in enumerate(doculects):
        for corres, count in correspondences[doculect].items():
            if count >= min_count:
                A[i, corres2int[corres]] = count
    print("Matrix shape: {}".format(A.shape))
    A_original = A.copy()
    if binary:
        A = A.astype(np.bool_)
    return A, A_original


def tfidf_hierarchical(A, doculects, context):
    A = TfidfTransformer().fit_transform(A)
    dist = 1 - cosine_similarity(A)
    Z = linkage(dist, method='average')
    fig, ax = plt.subplots()
    dendrogram(
        Z,
        labels=doculects,
        orientation='right',
        leaf_font_size=12.)
    fig.savefig('output/dendrogram-{}.pdf'.format(context),
                bbox_inches='tight')
    # Add new cluster IDs.
    n_samples = len(doculects)
    cluster_ids = np.arange(n_samples, n_samples + Z.shape[0]) \
                    .reshape(-1, 1)
    Z = np.hstack((Z, cluster_ids))
    with open('output/dendrogram-{}.pickle', 'w', encoding='utf8') as f:
        pickle.dump(Z, f)
    cluster2docs = {i: [d] for i, d in enumerate(doculects)}
    for row in Z:
        cluster2docs[row[-1]] = cluster2docs[row[0]] + cluster2docs[row[1]]
    clusters_and_doculects = [(c, d) for c, docs in cluster2docs.items()
                              for d in docs]
    return len(cluster2docs), clusters_and_doculects


def bsgc(A, k, doculects, all_correspondences, context):
    n_samples = len(doculects)
    n_features = len(all_correspondences)
    # Form the normalized matrix A_n.
    # Note that I already raise D_1, D_2 to the power of -0.5.
    D_1 = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        D_1[i, i] = np.sum(A[i])
    D_1 = linalg.sqrtm(np.linalg.inv(D_1))

    D_2 = np.zeros((n_features, n_features))
    for j in range(n_features):
        col_sum = np.sum(A, axis=0)
        if len(col_sum.shape) == 2 and col_sum.shape[0] == 1:
            # TODO delete this? seems unnecessary now
            # Otherwise, this clashes with the tf-idf transformed matrix.
            col_sum = np.array(col_sum).flatten()
        D_2[j, j] = col_sum[j]
    D_2 = linalg.sqrtm(np.linalg.inv(D_2))

    A_n = D_1 @ A @ D_2

    # Get the singular vectors of A_n.
    U, S, V_T = np.linalg.svd(A_n)
    V = np.transpose(V_T)

    # Use the singular vectors to get the eigenvectors.
    n_eigenvecs = math.ceil(math.log(k, 2))
    if args.verbose > 1:
        print("{} eigenvectors".format(n_eigenvecs))

    Z = np.zeros((n_samples + n_features, n_eigenvecs))
    Z[:n_samples] = D_1 @ U[:, 1:n_eigenvecs + 1]
    Z[n_samples:] = D_2 @ V[:, 1:n_eigenvecs + 1]
    if n_eigenvecs > 1:
        visualize(Z[:n_samples, 0], Z[:n_samples, 1], doculects, context)

    kmeans = cluster.KMeans(k)
    clusters = kmeans.fit_predict(Z)

    clusters_and_doculects = list(zip(clusters[:n_samples], doculects))
    clusters_and_features = list(zip(clusters[n_samples:],
                                     all_correspondences))
    return clusters_and_doculects, clusters_and_features


def score(A, corres, cluster_docs):
    cluster_size = len(cluster_docs)
    if cluster_size == 0:
        return 0, 0, 0, 0, 0
    # TODO currently binary
    occ_binary = 0
    occ_abs = 0
    total = 0
    for i in cluster_docs:
        if A[i, corres] > 0:
            occ_binary += 1
            occ_abs += A[i, corres]
        total += A[i].sum()
    rep = occ_binary / cluster_size
    rel_occ = occ_binary / np.sum(A[:, corres] > 0)
    rel_size = cluster_size / A.shape[0]
    dist = (rel_occ - rel_size) / (1 - rel_size)

    return rep, dist, (rep + dist) / 2, occ_abs / total, occ_abs


def visualize(x, y, labels, context):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]))
    fig.savefig('output/scatter-{}.pdf'.format(context), bbox_inches='tight')
    with open('output/scatter-{}.txt'.format(context),
              'w', encoding='utf8') as f:
        f.write("Place,x,y")
        for x_i, y_i, label in zip(x, y, labels):
            f.write("{},{},{}\n".format(label, x_i, y_i))


def print_clusters(A_original, k, clusters_and_doculects,
                   doculect2int, corres2int, corres2lang2word,
                   doculects, all_correspondences,
                   clusters_and_features=None):
    for c in range(k):
        print("\nCluster {}:\n-------------------------------------".format(c))
        ds = []
        for cl, d in clusters_and_doculects:
            if c == cl:
                print(d)
                ds.append(doculect2int[d])
        if clusters_and_features:
            fs = []
            for cl, f in clusters_and_features:
                if c == cl:
                    rep, dist, imp, rel, abs_n = score(A_original,
                                                       corres2int[f], ds)
                    fs.append((imp * 100, rep * 100, dist * 100,
                               rel * 100, abs_n, f))
        else:
            fs = []
            for f in all_correspondences:
                rep, dist, imp, rel, abs_n = score(A_original,
                                                   corres2int[f], ds)
                if imp > 0:
                    fs.append((imp * 100, rep * 100, dist * 100,
                               rel * 100, abs_n, f))
        fs = sorted(fs, reverse=True)
        print('-------')
        for j, (i, r, d, rel, a, f) in enumerate(fs):
            if j > 10:
                print("and {} more".format(len(fs) - j))
                break
            print("{}\t{:4.2f}\t(rep: {:4.2f}, dist: {:4.2f})\t{} ({:4.4f})"
                  .format(f, i, r, d, a, rel))
            for d in ds:
                try:
                    print(doculects[d], corres2lang2word[f][doculects[d]])
                except KeyError:
                    pass
            if len(ds) == 0:
                # Bipartite spectral graph clustering: clusters can consist of
                # only correspondences.
                print("{} doculect(s): {}".format(len(corres2lang2word[f]),
                                                  corres2lang2word[f]))
            print()
        print('=====================================')


if __name__ == "__main__":
    correspondences, all_correspondences, doculects, corres2lang2word = align(
        no_context=True, context_cv=True, context_sc=False,
        min_count=0, alignment_type='lib', alignment_mode='global',
        verbose=1)

    corres_no_context = [c for c in all_correspondences if len(c[0]) == 1]
    doculect2int = {x: i for i, x in enumerate(doculects)}

    # TODO construct A multiple times (with/without context, minimum count)
    # TODO call the tfidf & bsgc functions

    # print_clusters(A_original, k, clusters_and_doculects,
    #                doculect2int, corres2int, corres2lang2word,
    #                doculects, all_correspondences,
    #                clusters_and_features)
