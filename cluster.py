from align import align
from bsgc import bsgc_hierarchical
from scipy.cluster.hierarchy import dendrogram, linkage
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
    for k in clusters:
        print(k[0], len(k[1]))
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
