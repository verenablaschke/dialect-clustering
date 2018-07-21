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
import argparse
import sys


def score(A, corres, cluster_docs):
    if len(cluster_docs) == 0:
        return 0, 0, 0
    # TODO currently binary
    occ = 0
    for i in cluster_docs:
        if A[i, corres] > 0:
            occ += 1
    rep = occ / len(cluster_docs)
    rel_occ = occ / np.sum(A[:, corres] > 0)
    rel_size = len(cluster_docs) / A.shape[0]
    dist = (rel_occ - rel_size) / (1 - rel_size)

    # TODO try out harmonic mean?
    return rep, dist, (rep + dist) / 2


def visualize(x, y, labels):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for i, label in enumerate(labels):
        ax.annotate(label, (x[i], y[i]))
    # fig.savefig('svd.pdf', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-k', '--n_clusters', type=int, default=7,
        help='The number of clusters.')
    parser.add_argument(
        '-s', '--svd', dest='co_clustering', action='store_true',
        help='Include dimensionality reduction via SVD and co-clustering of '
        'doculects and correspondences.')
    parser.add_argument(
        '-n', '--direct', dest='co_clustering', action='store_false',
        help='Clustering is performed using the occurrence matrix.')
    parser.add_argument(
        '-b', '--binary', dest='binary', action='store_true',
        help='Binary matrix (feature present/absent?).')
    parser.add_argument(
        '-c', '--count', dest='binary', action='store_false',
        help='Matrix stores numbers of feature occurrences.')
    parser.add_argument(
        '--context_cv', dest='context_cv', action='store_true',
        help='Include left and right context of sound segments (C vs V vs #).')
    parser.add_argument(
        '--exclude_contextless', dest='context_none', action='store_false',
        help='Exclude sound segments without context.')
    parser.add_argument(
        '--context_sc', dest='context_sc', action='store_true',
        help='Include left and right context of sound segments '
             '(sound classes).')
    parser.add_argument(
        '-m', '--mincount', type=int, default=0,
        help='Minimum count per sound correspondence and doculect.')
    parser.add_argument(
        '-t', '--tfidf', dest='tfidf', action='store_true',
        help='Transform occurrence matrix with TF-IDF.')
    parser.add_argument(
        '--alignment_type', default='lib', choices=['lib', 'prog'],
        help='The LingPy alignment type.')
    parser.add_argument(
        '--alignment_mode', default='global', choices=['global', 'dialign'],
        help='The LingPy alignment mode.')
    parser.add_argument(
        '--msa_doculects', default='all', choices=['all', 'de-nl'],
        help='The BDPA doculects to be used during multi-alignment.')
    parser.add_argument(
        '-v', '--verbose', type=int, default=1, choices=[0, 1, 2, 3])
    parser.set_defaults(co_clustering=True,
                        binary=True, tfidf=False, exclude_contextless=False,
                        context_cv=False, context_sc=False)
    args = parser.parse_args()

    k = args.n_clusters
    if args.verbose > 0:
        print("`python {}`".format(" ".join(sys.argv)))
        print("Clusters: {}".format(k))
        print("Co-clustering: {}".format(args.co_clustering))
        print("Features: binary: {}, min. count {}, TF-IDF: {}"
              .format(args.binary, args.mincount, args.tfidf))
        print("Context: none: {}, CV: {}, sound classes: {}"
              .format(not args.exclude_contextless, args.context_cv,
                      args.context_sc))
        print("Alignment: {} {} ({})".format(args.alignment_mode,
                                             args.alignment_type,
                                             args.msa_doculects))
        print()

    correspondences, all_correspondences, doculects = align(
        no_context=not args.exclude_contextless,
        context_cv=args.context_cv, context_sc=args.context_sc,
        alignment_type=args.alignment_type, min_count=args.mincount,
        alignment_mode=args.alignment_mode, binary=args.binary,
        verbose=args.verbose)

    doculect2int = {x: i for i, x in enumerate(doculects)}
    corres2int = {x: i for i, x in enumerate(all_correspondences)}
    n_samples = len(doculects)
    n_features = len(all_correspondences)
    # A = sparse.dok_matrix(n_samples, n_features))
    A = np.zeros((n_samples, n_features), dtype=np.bool_)
    for i, doculect in enumerate(doculects):
        for corres, count in correspondences[doculect].items():
            A[i, corres2int[corres]] = 1 if args.binary else count
    if args.verbose > 0:
        print("Matrix shape: {}".format(A.shape))

    if args.tfidf:
        # TODO check out args
        transformer = TfidfTransformer(smooth_idf=False)
        A = transformer.fit_transform(A)
        print(A.shape)
        # x = PCA(2).fit_transform(A.todense())
        # visualize(x[:, 0], x[:, 1], doculects)
        # dist = 1 - cosine_similarity(A)
        # dendrogram(
        #     linkage(dist, method='average'),
        #     labels=doculects,
        #     orientation='right',
        #     leaf_font_size=12.)
        # plt.show()


    if args.co_clustering:
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
                # Else, this clashes with the tf-idf transformed matrix.
                col_sum = np.array(col_sum).flatten()
            D_2[j, j] = col_sum[j]
        D_2 = linalg.sqrtm(np.linalg.inv(D_2))

        A_n = D_1 @ A @ D_2

        # Get the singular values, singular vectors of A_n.

        U, S, V_T = np.linalg.svd(A_n)
        V = np.transpose(V_T)

        # TODO change this to hierarchical clustering?
        # Use the singular vectors to get the eigenvectors.

        # Rounding (up) isn't mentioned, but it seems necessary to do slices.
        n_eigenvecs = math.ceil(math.log(k, 2))
        # l = int(round(math.log(k, 2)))
        if args.verbose > 1:
            print("{} eigenvectors".format(n_eigenvecs))

        Z = np.zeros((n_samples + n_features, n_eigenvecs))
        # Does the n_eigenvecs+1 in the paper take care of rounding up?
        # (see above)
        # Why are we ignoring the first column/row?
        Z[:n_samples] = D_1 @ U[:, 1:1 + n_eigenvecs]
        v_2 = V[:, 1:1 + n_eigenvecs]
        Z[n_samples:] = D_2 @ v_2
        if n_eigenvecs > 1:
            # TODO it would be neat to cluster the most important sound
            # correspondences as well
            visualize(Z[:n_samples, 0], Z[:n_samples, 1], doculects)
    else:
        Z = A

    kmeans = cluster.KMeans(k)
    clusters = kmeans.fit_predict(Z)

    clusters_and_doculects = list(zip(clusters[:n_samples], doculects))
    if args.co_clustering:
        clusters_and_features = list(zip(clusters[n_samples:],
                                         all_correspondences))

    for c in range(k):
        print("\nCluster {}:\n-------------------------------------".format(c))
        ds = []
        for cl, d in clusters_and_doculects:
            if c == cl:
                print(d)
                ds.append(doculect2int[d])
        if args.co_clustering:
            fs = []
            for cl, f in clusters_and_features:
                if c == cl:
                    rep, dist, imp = score(A, corres2int[f], ds)
                    fs.append((imp * 100, rep * 100, dist * 100, f))
        else:
            fs = []
            for f in all_correspondences:
                rep, dist, imp = score(A, corres2int[f], ds)
                if imp > 0:
                    fs.append((imp * 100, rep * 100, dist * 100, f))
        fs = sorted(fs, reverse=True)
        print('-------')
        for j, (i, r, d, f) in enumerate(fs):
            if i < 80:
                print("and {} more".format(len(fs) - j))
                break
            print("{}\t{:4.2f}\t(rep: {:4.2f}, dist: {:4.2f})"
                  .format(f, i, r, d))
        print('=====================================')
