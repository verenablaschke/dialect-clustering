from lingpy.align.multiple import Multiple
from read_data import get_samples, DOCULECTS_BDPA, DOCULECTS_BDPA_ALL
from collections import Counter
from scipy import sparse, linalg
from sklearn import cluster, feature_extraction
import math
import numpy as np
import argparse


def align_concept(doculects, reference_doculect='ProtoGermanic',
                  alignment_type='lib', alignment_mode='global', verbose=1):
    sequences = []
    labels = [reference_doculect]
    for doculect, word in doculects.items():
        if doculect == reference_doculect:
            sequences.insert(0, word)
        else:
            sequences.append(word)
            labels.append(doculect)
    assert len(sequences) == len(labels), ("The reference doculect needs to be"
                                           " in the dict of doculects")
    msa = Multiple(sequences, merge_geminates=True)
    if alignment_type == 'lib':
        msa.lib_align(mode=alignment_mode)
    else:
        msa.prog_align(mode=alignment_mode)
    alignments = msa.alm_matrix
    if verbose > 2:
        print(msa)
    # TODO swaps
    corres = {}
    for i in range(1, len(labels)):
        c = zip(alignments[0], alignments[i])
        c = Counter([x for x in c if x != ('-', '-')])
        corres[labels[i]] = c
    return corres


def align(reference_doculect='ProtoGermanic', doculects_bdpa=DOCULECTS_BDPA,
          binary=True, msa_doculects_bdpa=DOCULECTS_BDPA_ALL,
          alignment_type='lib', alignment_mode='global', min_count=0,
          verbose=1):
    if verbose > 0:
        print('Reading the data files.')
    entries, doculects_all = get_samples(doculects_bdpa=msa_doculects_bdpa)
    doculects_all.remove(reference_doculect)
    correspondences = {}
    all_correspondences = Counter()

    if verbose > 0:
        print('Aligning the entries.')
    for concept, doculects in entries.items():
        if verbose > 2:
            print(concept)
        corres = align_concept(doculects,
                               reference_doculect=reference_doculect,
                               alignment_type=alignment_type,
                               alignment_mode=alignment_mode,
                               verbose=verbose)
        for doculect, tallies in corres.items():
            if doculect in msa_doculects_bdpa \
               and doculect not in doculects_bdpa:
                try:
                    doculects_all.remove(doculect)
                except ValueError:
                    pass
                continue
            all_correspondences.update(tallies)
            try:
                correspondences[doculect].update(tallies)
            except KeyError:
                correspondences[doculect] = tallies

    if min_count or verbose > 2:
        all_correspondences_old = all_correspondences.keys()
        all_correspondences = Counter()

        for doculect, tallies in correspondences.items():
            if binary:
                for corres in all_correspondences_old:
                    try:
                        # TODO a better way of excluding rare correspondences?
                        if tallies[corres] < min_count:
                            del tallies[corres]
                    except KeyError:
                        pass
            all_correspondences.update(tallies)
            if verbose > 2:
                print(doculect)
                print(tallies)
                print()

    all_correspondences = sorted(all_correspondences.keys())
    try:
        # In case there are any null-to-null alignments because of a larger
        # set of doculects being used for the alignment than for the tallies.
        all_correspondences.remove(('-', '-'))
    except ValueError:
        pass
    if verbose > 1:
        print(all_correspondences)
    return correspondences, all_correspondences, doculects_all


def score(A, corres, doculects):
    if len(doculects) == 0:
        return 0
    # TODO currently binary
    occ = 0
    for i in doculects:
        if A[i, corres] > 0:
            occ += 1
    rep = occ / len(doculects)
    rel_occ = occ / np.sum(A[:, corres] > 0)
    rel_size = len(doculects) / A.shape[0]
    dist = (rel_occ - rel_size) / (1 - rel_size)

    return rep, dist, (rep + dist) / 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-k', '--n_clusters', type=int, default=5,
        help='The number of clusters.')
    parser.add_argument(
        '-d', '--doculects', default='de-nl', choices=['de-nl', 'all'],
        help='The BDPA doculects to be included.')
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
    parser.set_defaults(co_clustering=True, binary=True, tfidf=False)
    args = parser.parse_args()

    k = args.n_clusters
    if args.verbose > 0:
        print("Clusters: {}".format(k))
        print("Doculects: {}".format(args.doculects))
        print("Co-clustering: {}".format(args.co_clustering))
        print("Binary features: {} (min. count {})".format(args.binary,
                                                           args.mincount))
        print("TF-IDF: {}".format(args.tfidf))
        print("Alignment: {} {} ({})".format(args.alignment_mode,
                                             args.alignment_type,
                                             args.msa_doculects))
        print()

    doculects_lookup = {'de-nl': DOCULECTS_BDPA, 'all': DOCULECTS_BDPA_ALL}
    correspondences, all_correspondences, doculects = align(
        doculects_bdpa=doculects_lookup[args.doculects],
        alignment_type=args.alignment_type, min_count=args.mincount,
        alignment_mode=args.alignment_mode, binary=args.binary,
        msa_doculects_bdpa=doculects_lookup[args.msa_doculects],
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
        transformer = feature_extraction.text.TfidfTransformer(smooth_idf=False)
        A = transformer.fit_transform(A)#.todense()
        print(A.shape)

    if args.co_clustering:
        # Form the normalized matrix A_n.
        # NOTE that I already raise D_1, D_2 to the power of -0.5
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
            fs = sorted(fs, reverse=True)
            print('-------')
            print(len(fs))
            for i, r, d, f in fs:
                print("{}\t{:4.2f}\t(rep: {:4.2f} dist: {:4.2f})"
                      .format(f, i, r, d))
        print('=====================================')
