import numpy as np


def tuple2corres(tup):
    # (('V', 'r'), ('-', 'ʁ'))
    # -> Vr > ∅ʁ
    hist = ''.join(tup[0])
    cur = ''.join(tup[1])
    return '{} > {}'.format(hist, cur).replace('-', '∅')


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
    if rel_size == 1:
        dist = 0
    else:
        dist = (rel_occ - rel_size) / (1 - rel_size)
    imp = 2 * rep * dist / (rep + dist)
    if dist < 0:
        imp = 0

    return rep, dist, imp, occ_abs


def print_clusters(filename, A_original, clusters, doculect2int, corres2int,
                   corres2lang2word, doculects, all_correspondences,
                   feature_clusters=None):
    fo = open(filename, 'w', encoding='utf8')
    if feature_clusters is None:
        feature_clusters = [None] * len(clusters)
    for c, features in zip(clusters, feature_clusters):
        fo.write("\n")
        fo.write(", ".join(c) + "\n")
        ds = [doculect2int[doc] for doc in c]
        fs = []
        if features:
            for f in features:
                rep, dist, imp, abs_n = score(A_original, corres2int[f], ds)
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
            if (j > 10 and i < 100) or i < 80:
                fo.write("and {} more\n".format(len(fs) - j))
                break
            fo.write("{}\t{:4.2f}\t(rep: {:4.2f}, dist: {:4.2f})"
                     "\t({} times)\n".format(tuple2corres(f), i, r, d, a))
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
