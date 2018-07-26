from lingpy.align.multiple import Multiple
from lingpy.sequence.sound_classes import token2class
from read_data import get_samples
from collections import Counter
import numpy as np


def align_concept(doculects, doculects_cwg, f, corres2lang2word=None,
                  reference_doculect='ProtoGermanic',
                  alignment_type='lib', alignment_mode='global',
                  no_context=True, context_cv=False, context_sc=False,
                  verbose=1):
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
    if verbose > 3:
        for line, label in zip(alignments, labels):
            print("{:15s} {}".format(label, '\t'.join(line)))
        print()

    if verbose > 1 and msa.swap_check():
        print("SWAPS!")
        print(msa.swap_index)
        print()

    # Remove doculects that were only used for alignment.
    msa_cwg = []
    labels_cwg = []
    for line, label in zip(alignments, labels):
        if label in doculects_cwg or label == reference_doculect:
            labels_cwg.append(label)
            msa_cwg.append(line)

    # Remove all-gap columns that might remain.
    gap_cols = np.nonzero(np.all(np.array(msa_cwg) == '-', axis=0))[0]
    if len(gap_cols) > 0:
        if verbose > 3:
            print("Removing columns", gap_cols)
        gap_cols = sorted(gap_cols, reverse=True)
        for line in msa_cwg:
            for g in gap_cols:
                del line[g]
    for line, label in zip(msa_cwg, labels_cwg):
        f.write("{:15s} {}\n".format(label, '\t'.join(line)))
    f.write("\n")
    alignments = msa_cwg
    labels = labels_cwg

    if context_cv:
        ref = ['#'] + alignments[0] + ['#']
        ref_segments_cv = []
        for j in range(1, len(ref) - 1):
            r_prev = seg2class(ref[j - 1])
            r = ref[j]
            r_next = seg2class(ref[j + 1])
            ref_segments_cv.append((r_prev, r))
            ref_segments_cv.append((r, r_next))
    if context_sc:
        ref = ['#'] + alignments[0] + ['#']
        ref_segments_sc = []
        for j in range(1, len(ref) - 1):
            r_prev = seg2class(ref[j - 1], sca=True)
            r = ref[j]
            r_next = seg2class(ref[j + 1], sca=True)
            ref_segments_sc.append((r_prev, r))
            ref_segments_sc.append((r, r_next))

    corres = {}
    if corres2lang2word is None:
        corres2lang2word = {}
    for i in range(1, len(labels)):
        corres_i = Counter()

        if no_context:
            c = zip(alignments[0], alignments[i])
            corres_i.update([(tuple([x[0]]), tuple([x[1]])) for x in c])

        ref_segments = []
        sca_model = []
        if context_cv:
            ref_segments.append(ref_segments_cv)
            sca_model.append(False)
        if context_sc:
            ref_segments.append(ref_segments_sc)
            sca_model.append(True)
        for ref_segs, sca in zip(ref_segments, sca_model):
            cur = ['#'] + alignments[i] + ['#']
            cur_segments = []
            for j in range(1, len(ref) - 1):
                c_prev = seg2class(cur[j - 1], sca=sca)
                c = cur[j]
                c_next = seg2class(cur[j + 1], sca=sca)
                cur_segments.append((c_prev, c))
                cur_segments.append((c, c_next))
            c = zip(ref_segs, cur_segments)
            corres_i.update([x for x in c])

        d = labels[i]
        corres[d] = corres_i
        for c in corres_i:
            try:
                # Use doculects[d] instead of sequences[i] because LingPy
                # changes the contents of 'sequences'.
                corres2lang2word[c][d].append(doculects[d])
            except KeyError:
                try:
                    corres2lang2word[c][d] = [doculects[d]]
                except KeyError:
                    corres2lang2word[c] = {d: [doculects[d]]}
    return corres, corres2lang2word


def seg2class(segment, sca=False):
    if segment in ['#', '-']:
        return segment
    if sca:
        return token2class(segment, 'sca')
    cl = token2class(segment, 'dolgo')
    return 'V' if cl == 'V' else 'C'


def align(reference_doculect='ProtoGermanic',
          alignment_type='lib', alignment_mode='global', min_count=0,
          no_context=True, context_cv=False, context_sc=False,
          verbose=1):
    f = open('output/alignments.txt', 'w', encoding='utf8')
    if verbose > 0:
        print('Reading the data files.')
    entries, doculects_cwg, doculects_add = get_samples()
    doculects_cwg.remove(reference_doculect)
    correspondences = {}
    all_correspondences = Counter()

    if verbose > 0:
        print('Aligning the entries.')
    corres2lang2word = None
    for concept, doculects in entries.items():
        f.write("{}\n".format(concept))
        corres, corres2lang2word = align_concept(doculects, doculects_cwg, f,
            corres2lang2word=corres2lang2word,
            reference_doculect=reference_doculect,
            alignment_type=alignment_type,
            alignment_mode=alignment_mode,
            no_context=no_context, context_cv=context_cv,
            context_sc=context_sc,
            verbose=verbose)
        for doculect, tallies in corres.items():
            all_correspondences.update(tallies)
            try:
                correspondences[doculect].update(tallies)
            except KeyError:
                correspondences[doculect] = tallies
    f.close()

    if min_count or verbose > 3:
        all_correspondences_old = all_correspondences.keys()
        all_correspondences = Counter()

        for doculect, tallies in correspondences.items():
            for corres in all_correspondences_old:
                try:
                    if tallies[corres] < min_count:
                        del tallies[corres]
                except KeyError:
                    pass
            all_correspondences.update(tallies)
            if verbose > 3:
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
    if verbose > 2:
        print(corres2lang2word)
    return (correspondences, all_correspondences,
            doculects_cwg, corres2lang2word)


if __name__ == "__main__":
    align(verbose=3)
