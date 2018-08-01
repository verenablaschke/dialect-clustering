import numpy as np


THRESHOLD = 80


def tuple2corres(tup):
    # (('V', 'r'), ('-', 'ʁ'))
    # -> Vr > ∅ʁ
    # hist = ''.join(tup[0])
    # cur = ''.join(tup[1])
    s = '{} > {}'.format(tup[0], tup[1])
    try:
        s += ' / {}'.format(tup[2])
    except IndexError:
        pass
    return s.replace('-', '∅')


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

    return 100 * rep, 100 * dist, 100 * imp, occ_abs


def print_clusters(filename, A_original, clusters, doculect2int, corres2int,
                   corres2lang2word, doculects, all_correspondences,
                   feature_clusters=None):
    n_above_threshold_total = 0
    importance_scores = []
    n_nonsingleton_notall = 0
    fo = open(filename, 'w', encoding='utf8')
    fo2 = open(filename.replace('.txt', '-complete.txt'), 'w', encoding='utf8')
    if feature_clusters is None:
        feature_clusters = [None] * len(clusters)
    for c, features in zip(clusters, feature_clusters):
        msg = "\n" + ", ".join(c) + "\n"
        fo.write(msg)
        fo2.write(msg)
        ds = [doculect2int[doc] for doc in c]
        if len(ds) > 1 and len(ds) < len(doculects):
            n_nonsingleton_notall += 1
        fs = []
        n_above_threshold = 0
        if features:
            for f in features:
                rep, dist, imp, abs_n = score(A_original, corres2int[f], ds)
                fs.append((imp, rep, dist, abs_n, f))
                if imp >= THRESHOLD:
                    n_above_threshold += 1
                    importance_scores.append(imp)
        else:
            for f in all_correspondences:
                rep, dist, imp, abs_n = score(A_original,
                                              corres2int[f], ds)
                if imp > 0 or (rep > 0.8 and len(ds) == len(doculects)):
                    fs.append((imp, rep, dist, abs_n, f))
                    if imp >= THRESHOLD:
                        n_above_threshold += 1
                        importance_scores.append(imp)
        n_above_threshold_total += n_above_threshold
        fs = sorted(fs, reverse=True)
        msg = "-------\n" \
              "{} correspondences above the threshold ({}% importance)\n\n" \
              .format(n_above_threshold, THRESHOLD)
        fo.write(msg)
        fo2.write(msg)
        wrote_more = False
        for j, (i, r, d, a, f) in enumerate(fs):
            msg = "{}\t{:4.2f}\t(rep: {:4.2f}, dist: {:4.2f})" \
                  "\t({} times)\n".format(tuple2corres(f), i, r, d, a)
            cont = (j < 10 or i == 100) and i >= THRESHOLD
            if (not cont) and (not wrote_more):
                fo.write("and {} more\n".format(len(fs) - j))
                wrote_more = True
                if not features:
                    break
            if cont:
                fo.write(msg)
            fo2.write(msg)
            for d in ds:
                try:
                    msg = "{}: {}\n".format(doculects[d],
                                            corres2lang2word[f][doculects[d]])
                    if cont:
                        fo.write(msg)
                    fo2.write(msg)
                except KeyError:
                    pass
            if len(ds) == 0:
                # Bipartite spectral graph clustering: clusters can consist of
                # only correspondences.
                msg = "{} doculect(s): {}\n".format(len(corres2lang2word[f]),
                                                    corres2lang2word[f])
                if cont:
                    fo.write(msg)
                fo2.write(msg)
            if cont:
                fo.write("\n")
            fo2.write("\n")
        fo.write("=====================================\n")
        fo2.write("=====================================\n")
    fo.write("\n\n{} clusters\n".format(len(clusters)))
    fo.write("{} clusters excl. singletons "
             "and the cluster including all doculects\n"
             .format(n_nonsingleton_notall))
    importance_scores = np.array(importance_scores)
    fo.write("max. importance score: {:4.2f}%\n"
             .format(np.amax(importance_scores)))
    fo.write("{} correspondences (total)\n".format(len(corres2int)))
    fo.write("{} correspondences >= the threshold ({}% importance)\n"
             .format(n_above_threshold_total, THRESHOLD))
    # Assuming the threshold isn't changed to anything above 90%.
    fo.write("{} correspondences >= 90% importance\n"
             .format(len(np.where(importance_scores >= 90)[0])))
    fo.write("{} correspondences >= 95% importance\n"
             .format(len(np.where(importance_scores >= 95)[0])))
    fo.write("{} correspondences == 100% importance\n"
             .format(len(np.where(importance_scores >= 99.99999)[0])))
    fo.close()
    fo2.close()
