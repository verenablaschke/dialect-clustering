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
        if c == 0:
            rows[0].append(i)
            docs[0].append(d)
        else:
            rows[1].append(i)
            docs[1].append(d)
    cols = [[], []]
    corres = [[], []]
    for i, (c, f) in enumerate(clusters_and_features):
        if c == 0:
            cols[0].append(i)
            corres[0].append(f)
        else:
            cols[1].append(i)
            corres[1].append(f)

    # Occasionally, sound correspondences whose eigenvalues are close to the
    # kmeans decision boundary end up in the "wrong" cluster in that they do
    # not belong to any of the doculects in this cluster. If the doculects-to-
    # features matrix contains empty columns(/rows), it cannot be normalized
    # in bsgc(). To prevent this, we change the labels for such instances.
    for cur, other in ((0, 1), (1, 0)):
        if docs[cur] and corres[cur]:
            A_new = A[rows[cur]]
            A_new = A_new[:, cols[cur]]
            non0 = np.nonzero(A_new)
            cols_non0 = set(non0[1])
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


def bsgc_hierarchical(A, doculects, all_correspondences, clusters=None):
    clusters_and_doculects, clusters_and_features = bsgc(A, 2, doculects,
                                                         all_correspondences)
    new_features = split_features(A, doculects, all_correspondences,
                                  clusters_and_doculects,
                                  clusters_and_features)
    if clusters is None:
        clusters = {}
    for (A, docs, corres) in new_features:
        clusters[docs] = corres
        if len(docs) > 2:
            clusters = bsgc_hierarchical(A, docs, corres, clusters)
    return clusters


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
    A, A_original, corres2int, all_corres = construct_A(all_correspondences,
                                                        correspondences,
                                                        doculects,
                                                        binary=True)
    print("Clustering.")
    clusters = bsgc_hierarchical(A, doculects, all_corres)
    print("\n")
    for k, v in clusters.items():
        print(k, len(v))
    # print("Scoring.")
    # print_clusters("output/bsgc-context.txt", A_original, k,
    #                clusters_and_doculects, doculect2int, corres2int,
    #                corres2lang2word, doculects, all_corres,
    #                clusters_and_features)

    # print("\nConstructing features for bsgc-nocontext.")
    # A, A_original, corres2int, all_corres = construct_A(corres_no_context,
    #                                                     correspondences,
    #                                                     doculects,
    #                                                     binary=True)
    # print("Clustering.")
    # bsgc_hierarchical(A, doculects, all_corres, context='nocontext')
    # print("Scoring.")
    # print_clusters("output/bsgc-nocontext.txt", A_original, k,
    #                clusters_and_doculects, doculect2int, corres2int,
    #                corres2lang2word, doculects, all_corres,
    #                clusters_and_features)
