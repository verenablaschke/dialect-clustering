from align import align
from scipy import linalg
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import cluster
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math
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
    with open('output/dendrogram-{}.pickle'.format(context), 'wb') as f:
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
    # print("{} eigenvector(s)".format(n_eigenvecs))

    Z = np.zeros((n_samples + n_features, n_eigenvecs))
    Z[:n_samples] = D_1 @ U[:, 1:n_eigenvecs + 1]
    Z[n_samples:] = D_2 @ V[:, 1:n_eigenvecs + 1]
    if n_eigenvecs > 1:
        visualize(Z[:n_samples, 0], Z[:n_samples, 1], doculects, context)

    kmeans = cluster.KMeans(k)
    clusters = kmeans.fit_predict(Z)
    # print(sorted(list(zip(clusters[:n_samples], Z[:n_samples], doculects))))
    # print(sorted(list(zip(clusters[n_samples:], Z[n_samples:], all_correspondences))))
    clusters_and_doculects = list(zip(clusters[:n_samples], doculects))
    clusters_and_features = list(zip(clusters[n_samples:],
                                     all_correspondences))
    return clusters_and_doculects, clusters_and_features


def split_features(A, doculects, all_correspondences,
                   clusters_and_doculects, clusters_and_features):
    rows_0 = []
    rows_1 = []
    docs_0 = []
    docs_1 = []
    for i, (c, d) in enumerate(clusters_and_doculects):
        if c == 0:
            rows_0.append(i)
            docs_0.append(d)
        else:
            rows_1.append(i)
            docs_1.append(d)
    print(rows_0)
    print(docs_0)
    print(rows_1)
    print(docs_1)
    cols_0 = []
    cols_1 = []
    corres_0 = []
    corres_1 = []
    for i, (c, f) in enumerate(clusters_and_features):
        if c == 0:
            cols_0.append(i)
            corres_0.append(f)
        else:
            cols_1.append(i)
            corres_1.append(f)

    A_0 = None
    if docs_0 and corres_0:
        A_0 = A[rows_0]
        A_0 = A_0[:, cols_0]
        non0 = np.nonzero(A_0)
        cols = set(non0[1])
        rows = set(non0[0])
        print(0)
        rows_to_move = []
        docs_to_move = []
        for c in range(A_0.shape[0]):
            if c not in rows:
                print("!", c, docs_0[c])
                rows_to_move.append(rows_0[c])
                docs_to_move.append(docs_0[c])
        cols_to_move = []
        corres_to_move = []
        for c in range(A_0.shape[1]):
            if c not in cols:
                print("!", c, corres_0[c])
                cols_to_move.append(cols_0[c])
                corres_to_move.append(corres_0[c])
        for r in rows_to_move:
            rows_0.remove(r)
            rows_1.append(r)
        for d in docs_to_move:
            docs_0.remove(d)
            docs_1.append(d)
        for c in cols_to_move:
            cols_0.remove(c)
            cols_1.append(c)
        for c in corres_to_move:
            corres_0.remove(c)
            corres_1.append(c)

    A_1 = None
    if docs_1 and corres_1:
        A_1 = A[rows_1]
        A_1 = A_1[:, cols_1]
        non0 = np.nonzero(A_1)
        cols = set(non0[1])
        rows = set(non0[0])
        print(1)
        rows_to_move = []
        docs_to_move = []
        for c in range(A_1.shape[0]):
            if c not in rows:
                print("!", c, docs_1[c])
                rows_to_move.append(rows_1[c])
                docs_to_move.append(docs_1[c])
        cols_to_move = []
        corres_to_move = []
        for c in range(A_1.shape[1]):
            if c not in cols:
                print("!", c, corres_1[c])
                cols_to_move.append(cols_1[c])
                corres_to_move.append(corres_1[c])
        for r in rows_to_move:
            rows_1.remove(r)
            rows_0.append(r)
        for d in docs_to_move:
            docs_1.remove(d)
            docs_0.append(d)
        for c in cols_to_move:
            cols_1.remove(c)
            cols_0.append(c)
        for c in corres_to_move:
            corres_1.remove(c)
            corres_0.append(c)

        A_0 = A[rows_0]
        A_0 = A_0[:, cols_0]
        A_1 = A[rows_1]
        A_1 = A_1[:, cols_1]
        print(A_0.shape, A_1.shape)
    return (A_0, docs_0, corres_0), (A_1, docs_1, corres_1)


def bsgc_hierarchical(A, doculects, all_correspondences, context):
    clusters_and_doculects, clusters_and_features = bsgc(A, 2, doculects,
                                                         all_correspondences,
                                                         context)
    print(0, len([x for x in clusters_and_features if x[0] == 0]), 'correspondences')
    print(0, len([x for x in clusters_and_doculects if x[0] == 0]), 'doculects')
    print(1, len([x for x in clusters_and_features if x[0] == 1]), 'correspondences')
    print(1, len([x for x in clusters_and_doculects if x[0] == 1]), 'doculects')
    new_features = split_features(A, doculects, all_correspondences, clusters_and_doculects, clusters_and_features)
    for (A, docs, corres) in new_features:
        if len(docs) > 2:
            bsgc_hierarchical(A, docs, corres, context)


def score(A, corres, cluster_docs):
    cluster_size = len(cluster_docs)
    if cluster_size == 0:
        return 0, 0, 0, 0, 0
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
    imp = 2 * rep * dist / (rep + dist)
    if dist < 0:
        imp = 0

    return rep, dist, imp, occ_abs


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


def print_clusters(filename, A_original, k, clusters_and_doculects,
                   doculect2int, corres2int, corres2lang2word,
                   doculects, all_correspondences,
                   clusters_and_features=None):
    fo = open(filename, 'w', encoding='utf8')
    for c in range(k):
        fo.write("\nCluster {}:\n--------------------------------\n".format(c))
        ds = []
        for cl, d in clusters_and_doculects:
            if c == cl:
                fo.write(d + "\n")
                ds.append(doculect2int[d])
        fs = []
        if clusters_and_features:
            for cl, f in clusters_and_features:
                if c == cl:
                    rep, dist, imp, abs_n = score(A_original,
                                                  corres2int[f], ds)
                    fs.append((imp * 100, rep * 100, dist * 100, abs_n, f))
        else:
            for f in all_correspondences:
                rep, dist, imp, abs_n = score(A_original,
                                              corres2int[f], ds)
                if imp > 0 or (rep > 0.9 and len(ds) == len(doculects)):
                    fs.append((imp * 100, rep * 100, dist * 100, abs_n, f))
        fs = sorted(fs, reverse=True)
        fo.write("-------\n")
        for j, (i, r, d, a, f) in enumerate(fs):
            if j > 10 and i < 100:
                fo.write("and {} more\n".format(len(fs) - j))
                break
            fo.write("{}\t{:4.2f}\t(rep: {:4.2f}, dist: {:4.2f})"
                     "\t({} times)\n".format(f, i, r, d, a))
            for d in ds:
                try:
                    fo.write("{}: {}\n"
                             .format(doculects[d],
                                     corres2lang2word[f][doculects[d]]))
                except KeyError:
                    pass
            if len(ds) == 0:
                # Bipartite spectral graph clustering: clusters can consist of
                # only correspondences.
                fo.write("{} doculect(s): {}\n"
                         .format(len(corres2lang2word[f]),
                                 corres2lang2word[f]))
            fo.write("\n")
        fo.write("=====================================\n")
    fo.close()


if __name__ == "__main__":
    correspondences, all_correspondences, doculects, corres2lang2word = align(
        no_context=True, context_cv=True, context_sc=True,
        min_count=3, alignment_type='lib', alignment_mode='global',
        verbose=1)

    corres_no_context = [c for c in all_correspondences if len(c[0]) == 1]
    doculect2int = {x: i for i, x in enumerate(doculects)}

    # print("Constructing features for tfidf-context.")
    # A, A_original, corres2int, all_corres = construct_A(all_correspondences,
    #                                                     correspondences,
    #                                                     doculects)
    # print("Creating dendrogram.")
    # k, clusters_and_doculects = tfidf_hierarchical(A, doculects,
    #                                                context='context')
    # print("Scoring.")
    # print_clusters("output/tfidf-context.txt", A_original, k,
    #                clusters_and_doculects, doculect2int, corres2int,
    #                corres2lang2word, doculects, all_corres)

    # print("\nConstructing features for tfidf-nocontext.")
    # A, A_original, corres2int, all_corres = construct_A(corres_no_context,
    #                                                     correspondences,
    #                                                     doculects)
    # print("Creating dendrogram.")
    # k, clusters_and_doculects = tfidf_hierarchical(A, doculects,
    #                                                context='nocontext')
    # print("Scoring.")
    # print_clusters("output/tfidf-nocontext.txt", A_original, k,
    #                clusters_and_doculects, doculect2int, corres2int,
    #                corres2lang2word, doculects, all_corres)

    # k = 5
    # print("\nConstructing features for bsgc-context.")
    # A, A_original, corres2int, all_corres = construct_A(all_correspondences,
    #                                                     correspondences,
    #                                                     doculects,
    #                                                     binary=True)
    # print("Clustering.")
    # clusters_and_doculects, clusters_and_features = bsgc(A, k, doculects,
    #                                                      all_corres,
    #                                                      context='context')
    # print("Scoring.")
    # print_clusters("output/bsgc-context.txt", A_original, k,
    #                clusters_and_doculects, doculect2int, corres2int,
    #                corres2lang2word, doculects, all_corres,
    #                clusters_and_features)

    print("\nConstructing features for bsgc-nocontext.")
    A, A_original, corres2int, all_corres = construct_A(corres_no_context,
                                                        correspondences,
                                                        doculects,
                                                        binary=True)
    print("Clustering.")
    bsgc_hierarchical(A, doculects, all_corres, context='nocontext')
    # print("Scoring.")
    # print_clusters("output/bsgc-nocontext.txt", A_original, k,
    #                clusters_and_doculects, doculect2int, corres2int,
    #                corres2lang2word, doculects, all_corres,
    #                clusters_and_features)
