from lingpy.align.multiple import Multiple
from read_data import get_samples
from collections import Counter
from scipy import sparse, linalg
from sklearn import cluster
import math
import numpy as np


def align_concept(doculects, reference_doculect='ProtoGermanic', verbose=1):
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
    # TODO this is doing quite badly with P-G suffixes
    msa.prog_align()
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


def align(reference_doculect='ProtoGermanic', verbose=1):
    if verbose > 0:
        print('Reading the data files.')
    entries, doculects_all = get_samples()
    doculects_all.remove(reference_doculect)
    correspondences = {}
    all_correspondences = Counter()

    if verbose > 0:
        print('Aligning the entries.')
    for concept, doculects in entries.items():
        if verbose > 2:
            print(concept)
        corres = align_concept(doculects, verbose=verbose)
        for doculect, tallies in corres.items():
            all_correspondences.update(tallies)
            try:
                correspondences[doculect].update(tallies)
            except KeyError:
                correspondences[doculect] = tallies

    all_correspondences_old = all_correspondences.keys()
    all_correspondences = Counter()

    for doculect, tallies in correspondences.items():
        if verbose > 2:
            print(doculect)
            print(tallies)
            print()
        for corres in all_correspondences_old:
            try:
                # TODO a better way of excluding rare correspondences?
                if tallies[corres] < 3:
                    del tallies[corres]
            except KeyError:
                pass
        all_correspondences.update(tallies)
        if verbose > 1:
            print(doculect)
            print(tallies)
            print()

    all_correspondences = sorted(all_correspondences.keys())
    return correspondences, all_correspondences, doculects_all


correspondences, all_correspondences, doculects = align()

# A = sparse.dok_matrix((len(doculects), len(all_correspondences)))
A = np.zeros((len(doculects), len(all_correspondences)), dtype=np.bool_)
for i, doculect in enumerate(doculects):
    for corres, count in correspondences[doculect].items():
        A[i, all_correspondences.index(corres)] = 1
print(A.shape)

# Form the normalized matrix A_n.
# NOTE that I already raise D_1, D_2 to the power of -0.5
n_samples, n_features = A.shape
D_1 = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    D_1[i, i] = np.sum(A[i])
D_1 = linalg.sqrtm(np.linalg.inv(D_1))

D_2 = np.zeros((n_features, n_features))
for j in range(n_features):
    D_2[j, j] = np.sum(A, axis=0)[j]
D_2 = linalg.sqrtm(np.linalg.inv(D_2))

A_n = D_1 @ A @ D_2

# Get the singular values, singular vectors of A_n.

U, S, V_T = np.linalg.svd(A_n)
V = np.transpose(V_T)

# TODO change this to hierarchical clustering?
# Use the singular vectors to get the eigenvectors.

k = 5  # number of clusters

# Rounding (up) isn't mentioned, but it seems necessary to get proper slices.)
n_eigenvecs = math.ceil(math.log(k, 2))
# l = int(round(math.log(k, 2)))
print(n_eigenvecs)

Z = np.zeros((n_samples + n_features, n_eigenvecs))
# Does the n_eigenvecs+1 in the paper take care of rounding up? (see above.)
# Why are we ignoring the first column/row?
Z[:n_samples] = D_1 @ U[:, 1:1 + n_eigenvecs]
v_2 = V[:, 1:1 + n_eigenvecs]
Z[n_samples:] = D_2 @ v_2
kmeans = cluster.KMeans(k)
clusters = kmeans.fit_predict(Z)

clusters_and_doculects = list(zip(clusters[:n_samples], doculects))
clusters_and_features = sorted(zip(v_2,
                                   clusters[n_samples:],
                                   all_correspondences),
                               reverse=True, key=lambda elem: elem[0][0])

for c in range(k):
    print("\nCluster {}:\n----------------------------------------".format(c))
    for cl, d in clusters_and_doculects:
        if c == cl:
            print(d)
    print('-------')
    for v, cl, f in clusters_and_features:
        if c == cl:
            print("{}\t{}".format(f, v))
    print('========================================')
