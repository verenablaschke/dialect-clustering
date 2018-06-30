from lingpy.align.multiple import Multiple
from read_data import get_samples
from collections import Counter


def align(doculects, reference_doculect='ProtoGermanic'):
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
    msa = Multiple(sequences)
    # TODO this is doing quite badly with P-G suffixes
    msa.prog_align()
    alignments = msa.alm_matrix
    # TODO swaps
    corres = {}
    for i in range(1, len(labels)):
        c = zip(alignments[0], alignments[i])
        c = Counter([x for x in c if x != ('-', '-')])
        corres[labels[i]] = c
    return corres


entries, doculects = get_samples()
correspondences = {}

for concept, doculects in entries.items():
    print(concept)
    corres = align(doculects)
    for doculect, tallies in corres.items():
        try:
            correspondences[doculect].update(tallies)
        except KeyError:
            correspondences[doculect] = tallies

for doculect, tallies in correspondences:
    print(doculect)
    print(tallies)
    print()
