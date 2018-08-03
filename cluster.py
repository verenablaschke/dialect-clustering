#!/usr/bin/env python3
from align import align
from bsgc import bsgc_hierarchical
from print_output import print_clusters
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import pickle


def construct_A(all_correspondences, correspondences, doculects,
                min_count=0, binary=False):
    # Remove rare correspondences that don't meet the min_count requirement.
    corres2int = {}
    i = 0
    for d in correspondences.values():
        for k, v in d.items():
            if k in corres2int:
                continue
            if k not in all_correspondences:
                continue
            if v >= min_count:
                corres2int[k] = i
                i += 1

    n_samples = len(doculects)
    n_features = len(corres2int)
    A = np.zeros((n_samples, n_features), dtype=np.int16)
    for i, doculect in enumerate(doculects):
        for corres, count in correspondences[doculect].items():
            if corres in all_correspondences and count >= min_count:
                A[i, corres2int[corres]] = count

    print("Matrix shape: {}".format(A.shape))
    A_original = A.copy()
    if binary:
        A = A.astype(np.bool_)

    if min_count == 0:
        all_corres = all_correspondences
    else:
        all_corres = sorted(corres2int.items(), key=lambda x: x[1])
        all_corres = [k for (k, v) in all_corres]
    return A, A_original, corres2int, all_corres


def tfidf_hierarchical(A, doculects, context):
    # TF-IDF
    A = TfidfTransformer().fit_transform(A)

    # Truncated SVD
    n_components = A.shape[0] if A.shape[0] < A.shape[1] else A.shape[1]
    svd = TruncatedSVD(n_components=n_components, random_state=123).fit(A)
    for i, v in enumerate(np.cumsum(svd.explained_variance_ratio_)):
        if v >= 0.85:
            n_components = i + 1
            break
    A = TruncatedSVD(n_components=n_components, random_state=123) \
        .fit_transform(A)
    print("Truncated A to {} compontents "
          "which explain {:4.2f}% of the variance".format(n_components,
                                                          v * 100))

    # Cluster via UPGMA and cosine similarity.
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

    # Update cluster IDs.
    n_samples = len(doculects)
    cluster_ids = np.arange(n_samples, n_samples + Z.shape[0]) \
                    .reshape(-1, 1)
    Z = np.hstack((Z, cluster_ids))
    with open('output/dendrogram-{}.pickle'.format(context), 'wb') as f:
        pickle.dump(Z, f)
    cluster2docs = {i: [d] for i, d in enumerate(doculects)}
    for row in Z:
        cluster2docs[row[-1]] = cluster2docs[row[0]] + cluster2docs[row[1]]
    clusters_and_doculects = [docs for _, docs in cluster2docs.items()]
    return clusters_and_doculects


if __name__ == "__main__":
    correspondences, all_correspondences, doculects, corres2lang2word = align(
        no_context=True, context_cv=True, context_sc=True, min_count=3,
        alignment_type='lib', alignment_mode='global', verbose=1)

    corres_no_context = [c for c in all_correspondences if len(c) == 2]
    doculect2int = {x: i for i, x in enumerate(doculects)}

    print("Constructing features for tfidf-context.")
    A, A_original, corres2int, all_corres = construct_A(all_correspondences,
                                                        correspondences,
                                                        doculects)
    print("Creating dendrogram.")
    clusters = tfidf_hierarchical(A, doculects, context='context')
    print("Scoring.")
    print_clusters("output/tfidf-context.txt", A_original,
                   clusters, doculect2int, corres2int,
                   corres2lang2word, doculects, all_corres)

    print("\nConstructing features for tfidf-nocontext.")
    A, A_original, corres2int, all_corres = construct_A(corres_no_context,
                                                        correspondences,
                                                        doculects)
    print("Creating dendrogram.")
    clusters = tfidf_hierarchical(A, doculects, context='nocontext')
    print("Scoring.")
    print_clusters("output/tfidf-nocontext.txt", A_original,
                   clusters, doculect2int, corres2int,
                   corres2lang2word, doculects, all_corres)

    print("\nConstructing features for bsgc-context.")
    A, A_original, corres2int, all_corres = construct_A(all_correspondences,
                                                        correspondences,
                                                        doculects,
                                                        binary=True)
    print("Clustering.")
    clusters, feature_clusters = bsgc_hierarchical(A, doculects, all_corres)
    print("Scoring.")
    print_clusters("output/bsgc-context.txt", A_original,
                   clusters, doculect2int, corres2int,
                   corres2lang2word, doculects, all_corres,
                   feature_clusters)

    print("\nConstructing features for bsgc-nocontext.")
    A, A_original, corres2int, all_corres = construct_A(corres_no_context,
                                                        correspondences,
                                                        doculects,
                                                        binary=True)
    print("Clustering.")
    clusters, feature_clusters = bsgc_hierarchical(A, doculects, all_corres)
    print("Scoring.")
    print_clusters("output/bsgc-nocontext.txt", A_original,
                   clusters, doculect2int, corres2int,
                   corres2lang2word, doculects, all_corres,
                   feature_clusters)
